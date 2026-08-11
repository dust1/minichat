import os
import glob
import sentencepiece as spm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "train_dataset")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "out", "bbpe_tokenizer")

def compute_vocab_size(data_dir: str) -> int:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    total_bytes = 0
    for fpath in paths:
        with open(fpath, "rb") as f:
            total_bytes += len(f.read())
    # 建议对于测试或中小型语料，不要直接拉满 32000，词表越大训练越慢
    size = min(32000, max(2000, total_bytes // 10 + 1000))
    print(f"[INFO] Total data bytes: {total_bytes:,} -> vocab_size={size}")
    return size

def train_bbpe(input_dir: str, model_prefix: str, vocab_size: int):
    paths = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    if not paths:
        print(f"[ERROR] No .txt files found in {input_dir}")
        return

    print("开始训练，请耐心等待...")
    spm.SentencePieceTrainer.train(
        input=paths,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",

        # 特殊 token 的 ID 必须与 tokenizer.py / dataloader.py 中的常量一致
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,                    # pad 用于批次内 padding，训练时被 mask 掉

        # 【核心加速1】不要设为1.0！0.9995 是业界标准，丢弃极罕见字符，大幅加速
        character_coverage=0.9995,
        byte_fallback=True,          
        
        # 【核心加速2】显式指定多线程，根据你的CPU核心数调整，比如 8 或 16
        num_threads=os.cpu_count(), 
        
        hard_vocab_limit=False,
        normalization_rule_name="identity",
        split_by_unicode_script=False,
        split_by_whitespace=False,
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        
        # 【核心加速3】防止超长单行文本拖慢内存和解析
        max_sentence_length=8192,   
        
        # 【可选加速4】如果语料极大（>500MB），可开启采样限制，加速明显
        # input_sentence_size=2000000, 
        # shuffle_input_sentence=True, 
    )

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_prefix = os.path.join(OUTPUT_DIR, "bbpe")
    
    vocab_size = compute_vocab_size(DATA_DIR)
    print(f"[Step 1] Training SentencePiece BBPE directly (vocab_size={vocab_size})...")
    train_bbpe(DATA_DIR, model_prefix, vocab_size)
    
    print(f"[OK] BBPE tokenizer saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()