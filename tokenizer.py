import os
import glob
import json
import numpy as np
import sentencepiece as spm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "out", "bbpe_tokenizer", "bbpe.model")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "out", "tokenized_data")

# 特殊 token ID（与 token_train.py 的 unk_id/bos_id/eos_id/pad_id 保持一致）
UNK_ID, BOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3
SPECIAL_IDS = {UNK_ID, BOS_ID, EOS_ID, PAD_ID}

sp = spm.SentencePieceProcessor()
sp.Load(MODEL_PATH)


def load_tokenizer(model_path):
    """从指定路径加载 SentencePiece 处理器（如 checkpoint 随附的 bbpe.model）。

    重新绑定模块级 sp，使后续 encode/decode 使用该词表；返回处理器本身。
    """
    global sp
    p = spm.SentencePieceProcessor()
    p.Load(model_path)
    sp = p
    return sp


def vocab_size() -> int:
    return sp.GetPieceSize()


def encode(text: str) -> list[int]:
    # 与 token_train.py 的训练语料一致：直接对原始 Unicode 文本编码
    # （不再做字节->Latin-1 映射；空格由 SP 内部以 U+2581 ▁ 表示）
    return sp.EncodeAsIds(text)


def decode(ids) -> str:
    # 跳过特殊 token，逐 piece 还原：
    #  - 字节回退 piece <0xXX> 还原为原始字节
    #  - 其余 piece 按 UTF-8 文本输出
    # 最后统一把 U+2581(▁，SP 内部的空格标记) 还原为普通空格
    valid = [int(i) for i in ids if int(i) not in SPECIAL_IDS]
    if not valid:
        return ""
    out = bytearray()
    for pid in valid:
        p = sp.IdToPiece(pid)
        if p.startswith("<0x") and len(p) == 6:
            try:
                out.append(int(p[3:5], 16))
            except ValueError:
                out += p.encode("utf-8")
        else:
            out += p.encode("utf-8")
    return out.decode("utf-8", errors="replace").replace("\u2581", " ")


def tokenize_texts(data_dir: str, output_dir: str = OUTPUT_DIR, val_fraction: float = 0.1):
    os.makedirs(output_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not paths:
        print(f"[ERROR] No .txt files found in {data_dir}")
        return None, 0

    docs = []
    for fpath in paths:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        text = text.strip()
        if not text:
            continue
        ids = encode(text)
        if not ids:
            continue
        # 每个文档末尾追加 EOS，作为文档边界
        docs.append((os.path.basename(fpath), ids + [EOS_ID]))

    if not docs:
        print("[ERROR] No tokens generated from any file.")
        return None, 0

    # 按文档划分 train/val：文档少于 2 个时全部用于训练
    n_val = 0 if len(docs) < 2 else max(1, int(round(len(docs) * val_fraction)))
    train_docs = docs[:-n_val] if n_val else docs
    val_docs = docs[-n_val:] if n_val else []

    def _write(name, subset):
        all_ids = []
        offsets = []
        offset = 0
        for fname, ids in subset:
            all_ids.extend(ids)
            offsets.append({"file": fname, "start": offset, "end": offset + len(ids)})
            offset += len(ids)
        arr = np.array(all_ids, dtype=np.int32)
        arr.tofile(os.path.join(output_dir, name))
        return len(arr), offsets

    train_tokens, train_offsets = _write("train_data.bin", train_docs)
    val_tokens, val_offsets = _write("val_data.bin", val_docs)

    meta = {
        "vocab_size": sp.GetPieceSize(),
        "special_ids": {"unk": UNK_ID, "bos": BOS_ID, "eos": EOS_ID, "pad": PAD_ID},
        "tokenizer": "bbpe.model",
        "val_fraction": val_fraction,
        "num_files": len(docs),
        "train": {"total_tokens": train_tokens, "num_files": len(train_offsets), "file_offsets": train_offsets},
        "val": {"total_tokens": val_tokens, "num_files": len(val_offsets), "file_offsets": val_offsets},
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Tokenized {len(docs)} files -> train {train_tokens:,} tokens, val {val_tokens:,} tokens")
    print(f"     EOS(ID={EOS_ID}) inserted at each document end")
    print(f"     Vocab size: {sp.GetPieceSize()}")
    return os.path.join(output_dir, "train_data.bin"), sp.GetPieceSize()


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "train_dataset")
    output_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    tokenize_texts(data_dir, output_dir)
