# -*- coding: utf-8 -*-
"""将 MiniGPT checkpoint 导出为 Hugging Face / safetensors 格式（标准 LLaMA 架构）。

由于本模型在架构上与 LlamaForCausalLM 完全同构（RoPE + RMSNorm + SwiGLU + GQA +
tied embeddings + 无 bias/dropout，RoPE 采用与 HF 一致的 GPT-NeoX 约定），
导出后可直接被 transformers / vLLM / SGLang 加载，无需任何自定义代码。

导出目录内容：
  config.json             — LlamaConfig 兼容字段
  generation_config.json  — 采样默认参数
  model.safetensors       — 权重（tied embeddings 只存 embed_tokens 一份）
  tokenizer.model         — SentencePiece 模型（HF/vLLM 会自动转换为 fast tokenizer）
  tokenizer_config.json   — tokenizer 声明

用法：
  python export_hf.py --ckpt-dir model_weights --out-dir model_weights_hf

导出后服务（需要 Linux；vLLM/SGLang 不支持原生 Windows）：
  vllm serve model_weights_hf
  python -m sglang.launch_server --model-path model_weights_hf
"""

import argparse
import json
import os
import shutil

import torch

from model import ModelConfig

# 特殊 token ID（与 tokenizer.py / token_train.py 的约定保持一致）。
# 此处本地定义而不 import tokenizer，避免模块级加载 bbpe.model 的副作用。
UNK_ID, BOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3


# ========== 配置映射 ==========
def to_hf_config(cfg: ModelConfig) -> dict:
    """ModelConfig -> LlamaConfig 兼容字典。"""
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.d_ff,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "max_position_embeddings": cfg.max_len,
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": cfg.rms_norm_eps,
        "tie_word_embeddings": True,
        "bos_token_id": BOS_ID,
        "eos_token_id": EOS_ID,
        "pad_token_id": PAD_ID,
        "torch_dtype": "float32",
    }


def to_generation_config(temperature=0.8, top_k=50, top_p=0.95,
                         repetition_penalty=1.05) -> dict:
    """HF GenerationConfig 兼容的采样默认参数。"""
    return {
        "bos_token_id": BOS_ID,
        "eos_token_id": EOS_ID,
        "pad_token_id": PAD_ID,
        "do_sample": True,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }


# ========== 权重映射 ==========
def to_hf_state_dict(sd: dict, cfg: ModelConfig) -> dict:
    """MiniGPT state_dict -> HF LlamaForCausalLM state_dict。

    关键映射：
      - 融合 qkv_proj 按输出维拆分为 q/k/v_proj（与 forward 中 split 的顺序一致）
      - RoPE 采用 GPT-NeoX 约定（与 HF rotate_half 相同），无需维度置换
      - tied embeddings：lm_head.weight 不单独存储，HF 加载时按
        tie_word_embeddings=True 自动绑定到 embed_tokens
    """
    head_dim = cfg.d_model // cfg.n_heads
    q_rows = cfg.n_heads * head_dim
    kv_rows = cfg.n_kv_heads * head_dim

    out = {"model.embed_tokens.weight": sd["tok_emb.weight"]}
    for i in range(cfg.n_layers):
        src, dst = f"blocks.{i}", f"model.layers.{i}"
        qkv = sd[f"{src}.attn.qkv_proj.weight"]
        q, k, v = qkv.split([q_rows, kv_rows, kv_rows], dim=0)
        out[f"{dst}.self_attn.q_proj.weight"] = q.contiguous()
        out[f"{dst}.self_attn.k_proj.weight"] = k.contiguous()
        out[f"{dst}.self_attn.v_proj.weight"] = v.contiguous()
        out[f"{dst}.self_attn.o_proj.weight"] = sd[f"{src}.attn.out_proj.weight"]
        out[f"{dst}.mlp.gate_proj.weight"] = sd[f"{src}.ffn.gate_proj.weight"]
        out[f"{dst}.mlp.up_proj.weight"] = sd[f"{src}.ffn.up_proj.weight"]
        out[f"{dst}.mlp.down_proj.weight"] = sd[f"{src}.ffn.down_proj.weight"]
        out[f"{dst}.input_layernorm.weight"] = sd[f"{src}.norm1.weight"]
        out[f"{dst}.post_attention_layernorm.weight"] = sd[f"{src}.norm2.weight"]
    out["model.norm.weight"] = sd["norm_f.weight"]
    return out


def expected_hf_keys(cfg: ModelConfig) -> set:
    """LlamaForCausalLM（tied）期望的权重键集合，用于导出前自检。"""
    keys = {"model.embed_tokens.weight", "model.norm.weight"}
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}"
        keys |= {
            f"{p}.self_attn.q_proj.weight", f"{p}.self_attn.k_proj.weight",
            f"{p}.self_attn.v_proj.weight", f"{p}.self_attn.o_proj.weight",
            f"{p}.mlp.gate_proj.weight", f"{p}.mlp.up_proj.weight",
            f"{p}.mlp.down_proj.weight",
            f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight",
        }
    return keys


# ========== 导出 ==========
def export(ckpt_dir: str, out_dir: str, device: str = "cpu"):
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"checkpoint 目录不存在: {ckpt_dir}")
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise SystemExit("缺少依赖：pip install safetensors")
    # 延迟导入：train -> dataloader 会在模块级加载 bbpe.model，
    # 仅在真正导出时（tokenizer 已存在的环境）才需要
    from train import load_checkpoint

    model, ckpt = load_checkpoint(ckpt_dir, torch.device(device))
    cfg = model.cfg
    hf_sd = to_hf_state_dict(model.state_dict(), cfg)

    # 自检：键集合与 HF 期望完全一致
    missing = expected_hf_keys(cfg) - set(hf_sd)
    extra = set(hf_sd) - expected_hf_keys(cfg)
    assert not missing and not extra, f"键不匹配: missing={missing}, extra={extra}"

    os.makedirs(out_dir, exist_ok=True)
    save_file(hf_sd, os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt"})

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(to_hf_config(cfg), f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump(to_generation_config(), f, ensure_ascii=False, indent=2)

    # tokenizer：优先取 checkpoint 随附的 bbpe.model（与训练词表严格一致）
    tok_src = None
    for cand in (os.path.join(ckpt_dir, "bbpe.model"),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "out", "bbpe_tokenizer", "bbpe.model")):
        if os.path.isfile(cand):
            tok_src = cand
            break
    if tok_src is None:
        raise FileNotFoundError("找不到 bbpe.model（checkpoint 目录或 out/bbpe_tokenizer）")
    # HF 约定：SentencePiece 模型文件命名为 tokenizer.model
    shutil.copy2(tok_src, os.path.join(out_dir, "tokenizer.model"))
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "tokenizer_class": "LlamaTokenizer",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "model_max_length": cfg.max_len,
        }, f, ensure_ascii=False, indent=2)

    n_params = sum(t.numel() for t in hf_sd.values())
    print(f"[OK] 导出完成: {out_dir}")
    print(f"     参数: {n_params:,} (tied: lm_head 共享 embed_tokens)")
    print(f"     架构: LlamaForCausalLM  {json.dumps(to_hf_config(cfg), ensure_ascii=False)}")
    print(f"     服务: vllm serve {out_dir}   或   "
          f"python -m sglang.launch_server --model-path {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="导出 MiniGPT 为 HF/safetensors 格式")
    parser.add_argument("--ckpt-dir", default="./model_weights")
    parser.add_argument("--out-dir", default="./model_weights_hf")
    args = parser.parse_args()
    export(args.ckpt_dir, args.out_dir)


if __name__ == "__main__":
    main()
