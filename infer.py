# -*- coding: utf-8 -*-
"""MiniChat 推理入口（PyTorch 版，KV cache 增量解码）。

从 model_weights/ 加载 ckpt.pt（或 ckpt_last.pt）与随附的 tokenizer 生成文本。
预填充一次后逐 token 增量解码，无需每步重算全上下文。
如需高并发服务，请先运行 export_hf.py 导出 HF 格式，再用 vLLM/SGLang 部署。
"""

import argparse
import os

import torch

import tokenizer as tok
from model import generate_ids, pick_amp_dtype
from train import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="MiniChat inference (PyTorch)")
    parser.add_argument("prompt", type=str, nargs="?", default="从前有座山")
    parser.add_argument("--model-dir", default="./model_weights")
    parser.add_argument("--max-gen", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--rep-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"Error: model directory '{args.model_dir}' not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = pick_amp_dtype(device)
    print(f"Loading model... (device={device.type})")
    try:
        model, ckpt = load_checkpoint(args.model_dir, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    model.eval()
    print(f"Checkpoint loaded: step={ckpt.get('step')}, "
          f"best_val_loss={ckpt.get('best_val_loss')}")

    tok_path = os.path.join(args.model_dir, "bbpe.model")
    if not os.path.isfile(tok_path):
        print(f"Error: tokenizer file '{tok_path}' not found in model directory.")
        return
    tok.load_tokenizer(tok_path)
    if tok.vocab_size() != model.cfg.vocab_size:
        print(f"Error: tokenizer vocab size {tok.vocab_size()} != "
              f"model vocab size {model.cfg.vocab_size}; checkpoint 与 tokenizer 不匹配")
        return
    print(f"Tokenizer loaded from {tok_path} (vocab={tok.vocab_size()})")

    print("Generating...")
    prompt_ids = tok.encode(args.prompt)
    gen_ids = generate_ids(model, prompt_ids, max_new=args.max_gen,
                           temperature=args.temperature, top_k=args.top_k,
                           top_p=args.top_p, repetition_penalty=args.rep_penalty,
                           seed=args.seed, eos_id=tok.EOS_ID, amp_dtype=amp_dtype)

    print()
    print("Prompt   :", args.prompt)
    print("Generated:", tok.decode(gen_ids)[:500])


if __name__ == "__main__":
    main()
