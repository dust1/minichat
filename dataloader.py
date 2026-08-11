import os
import numpy as np
import sentencepiece as spm
from tokenizer import UNK_ID, BOS_ID, EOS_ID, PAD_ID

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "out", "bbpe_tokenizer", "bbpe.model")

IGNORE_INDEX = -1

_sp = spm.SentencePieceProcessor()
_sp.Load(MODEL_PATH)


class TokenDataset:
    def __init__(self, data_dir="", split="train", max_len=512, stride=256,
                 eos_id=EOS_ID, pad_id=PAD_ID, ignore_index=IGNORE_INDEX):
        if not data_dir:
            data_dir = os.path.join(SCRIPT_DIR, "out", "tokenized_data")
        self.max_len = max_len
        self.pad_id = pad_id
        self.ignore_index = ignore_index
        self.eos_id = eos_id
        self.vocab_size = _sp.GetPieceSize()
        self.tokens = np.empty(0, dtype=np.int32)
        self.windows = []

        bin_path = os.path.join(data_dir, "train_data.bin" if split == "train" else "val_data.bin")
        if not os.path.isfile(bin_path):
            if split == "val":
                return  # 空验证集，由调用方跳过
            raise FileNotFoundError(f"Binary file not found: {bin_path}. Run tokenizer.py first.")

        self.tokens = np.fromfile(bin_path, dtype=np.int32)
        self._build_windows(max_len, stride)

    def _build_windows(self, max_len, stride):
        # 以 EOS 为界切分文档（EOS 属于前一文档，作为其最后一个可预测 token）
        start = 0
        for i, tid in enumerate(self.tokens):
            if tid == self.eos_id:
                self._add_doc_windows(start, i + 1, max_len, stride)
                start = i + 1
        if start < len(self.tokens):
            self._add_doc_windows(start, len(self.tokens), max_len, stride)

    def _add_doc_windows(self, s, e, max_len, stride):
        # 窗口绝不跨文档；不足 max_len 的文档/尾部窗口在读取时 padding
        # 存储三元组 (window_start, window_end, doc_end) 以正确处理目标窗口
        if e <= s:
            return
        if e - s <= max_len:
            self.windows.append((s, e, e))
            return
        pos = s
        while pos + max_len < e:
            self.windows.append((pos, pos + max_len, e))
            pos += stride
        if pos < e:
            self.windows.append((pos, e, e))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        s, window_e, doc_e = self.windows[idx]
        x = self.tokens[s:window_e]
        if len(x) < self.max_len:
            x = np.concatenate([x, np.full(self.max_len - len(x), self.pad_id, dtype=np.int32)])
        # 目标窗口：从 s+1 开始，最多到 window_e（包含 window_e 处的 token）
        # 但不能超过文档边界 doc_e
        y = self.tokens[s + 1:min(window_e + 1, doc_e)]
        if len(y) < self.max_len:
            y = np.concatenate([y, np.full(self.max_len - len(y), self.ignore_index, dtype=np.int32)])
        return x.astype(np.int32, copy=False), y.astype(np.int32, copy=False)

    def get_batch(self, batch_indices):
        X, Y = [], []
        for i in batch_indices:
            x, y = self[i]
            X.append(x)
            Y.append(y)
        return np.array(X, dtype=np.int32), np.array(Y, dtype=np.int32)

    def prompt_ids(self, n=30):
        if not self.windows:
            return []
        s, e, _ = self.windows[0]
        return self.tokens[s:s + min(n, e - s)].tolist()


def create_dataloader(data_dir="./out/tokenized_data", split="train", max_len=512, stride=256):
    return TokenDataset(data_dir, split, max_len, stride)
