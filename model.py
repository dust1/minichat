# -*- coding: utf-8 -*-
"""MiniGPT：PyTorch 版 Decoder-only Transformer 模型定义（标准 LLaMA 形态）。

架构：RoPE + RMSNorm + SwiGLU + GQA + tied embeddings，全部 Linear 无 bias，无 dropout。

已完成批次：
  第一批 — PyTorch autograd / SDPA 多头注意力 / RMSNorm / SwiGLU /
           tied embeddings / BF16-FP16 混合精度 / TensorBoard 日志
  第二批 — RoPE（旋转位置编码）、GQA（分组查询注意力）、移除全部 bias / dropout
  第三批 — KV cache 增量推理、safetensors / Hugging Face 导出（见 export_hf.py）

后续批次计划：gradient checkpointing、FSDP 多卡训练。
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== 配置 ==========
@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    # GQA：KV 头数；n_kv_heads == n_heads 时退化为标准 MHA，为 1 时是 MQA
    n_kv_heads: int = 2
    n_layers: int = 4
    # SwiGLU 为三个矩阵（gate/up/down），取 176 时参数量与旧 d_ff=256 的
    # 两矩阵 ReLU FFN 基本相当：3*d*176 ≈ 2*d*256
    d_ff: int = 176
    max_len: int = 512
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model({self.d_model}) 必须能被 n_heads({self.n_heads}) 整除")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads({self.n_heads}) 必须能被 n_kv_heads({self.n_kv_heads}) 整除")


# ========== RoPE ==========
def build_rope_cache(head_dim: int, max_len: int, theta: float = 10000.0):
    """预计算 RoPE 的 cos/sin 缓存，形状 (max_len, head_dim)，fp32。

    采用 GPT-NeoX / HF-LLaMA 约定：维度一分为二，前半与后半成对旋转。
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)            # (max_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)     # (max_len, head_dim)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 (B, H, S, head_dim) 的 Q/K 施加旋转位置编码。

    cos/sin: (S, head_dim)。旋转是正交变换，不改变向量范数；
    位置 i 的 query 与位置 j 的 key 的内积只依赖相对距离 (i-j)。
    """
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# ========== 基础层 ==========
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization（无 mean 中心化、无 bias）。

    均方根在 fp32 下计算，保证 bf16/fp16 混合精度训练的数值稳定性。
    等价于 torch.nn.RMSNorm（PyTorch 2.4+），此处手写以兼容旧版本。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * x).to(input_dtype)


class CausalSelfAttention(nn.Module):
    """GQA 因果自注意力 + RoPE，核心计算由 SDPA 完成。

    SDPA 会按硬件自动选择 flash / memory-efficient / math 内核，
    注意力部分显存从 O(S^2) 降为 O(S)（flash/mem-efficient 后端）。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads  # 每个 KV 头服务的 Q 头数
        # 融合的 QKV 投影：Q 为 n_heads 份，K/V 各为 n_kv_heads 份
        self.qkv_proj = nn.Linear(
            cfg.d_model,
            (cfg.n_heads + 2 * cfg.n_kv_heads) * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                kv_cache=None, use_cache: bool = False):
        """x: (B, S, D)；kv_cache: (k_past, v_past)，各为 (B, n_kv_heads, S_past, head_dim)。

        返回 (out, new_cache)；use_cache=False 时 new_cache 为 None。
        """
        B, S, D = x.shape
        q, k, v = self.qkv_proj(x).split(
            [self.n_heads * self.head_dim,
             self.n_kv_heads * self.head_dim,
             self.n_kv_heads * self.head_dim], dim=-1)
        # (B, S, *) -> (B, H, S, head_dim)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # RoPE 只作用于 Q/K，V 不携带位置信息；cache 存广播前的 KV 头（GQA 省显存的关键）
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if kv_cache is not None:
            k = torch.cat((kv_cache[0], k), dim=2)
            v = torch.cat((kv_cache[1], v), dim=2)
        new_cache = (k, v) if use_cache else None
        # GQA：将 KV 头广播到 Q 头数（n_rep=1 时退化为 MHA，无开销）。
        # PyTorch 2.5+ 可用 SDPA 的 enable_gqa=True 避免显式广播，此处为兼容旧版本手动展开。
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        # 因果 mask：仅在无缓存的全量前向（q_len == kv_len > 1）时使用 is_causal；
        # 逐 token 解码（q_len=1）时所有历史 token 均可见，无需 mask
        # （q_len != kv_len 时 is_causal 的 mask 对齐方式不适用于缓存解码，故必须区分）。
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=(kv_cache is None and q.size(2) > 1))
        # (B, H, S, head_dim) -> (B, S, D)
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(y), new_cache


class SwiGLU(nn.Module):
    """SwiGLU FFN：down(silu(gate(x)) * up(x))，三个无 bias 矩阵。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Pre-Norm Transformer Block：RMSNorm → Attention → 残差；RMSNorm → SwiGLU → 残差。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                kv_cache=None, use_cache: bool = False):
        attn_out, new_cache = self.attn(
            self.norm1(x), cos, sin, kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


# ========== 主模型 ==========
class MiniGPT(nn.Module):
    """标准 LLaMA 形态的 Decoder-only 语言模型。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # tied embeddings：输出头与输入 embedding 共享同一权重矩阵。
        # 对 embedding 占比很高的小模型可显著省参数，同时起到正则化作用。
        self.lm_head.weight = self.tok_emb.weight

        # RoPE cos/sin 缓存：非持久化 buffer，不进 state_dict，随 .to(device) 移动
        head_dim = cfg.d_model // cfg.n_heads
        cos, sin = build_rope_cache(head_dim, cfg.max_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # 残差分支的输出投影按 GPT-2 惯例缩小初始化，稳定深层堆叠训练
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                start_pos: int = 0, kv_cache=None, use_cache: bool = False):
        """idx: (B, S) int64；targets: (B, S) int64，无效位置为 -1。

        训练时返回 (None, loss)。
        推理时返回 (logits[:, -1:], caches)：只计算最后一个位置的 logits，
        避免物化无用的 (B, S, V)；use_cache=False 时 caches 为 None。
        kv_cache: 每层一个 (k, v) 元组的列表（增量解码）；start_pos 为 RoPE 位置偏移。
        """
        B, S = idx.shape
        if kv_cache is None and start_pos == 0 and S > self.cfg.max_len:
            # 无缓存全量前向：右截断（与旧行为一致）
            idx = idx[:, : self.cfg.max_len]
            S = self.cfg.max_len
            if targets is not None:
                targets = targets[:, : self.cfg.max_len]
        if start_pos + S > self.cfg.max_len:
            raise ValueError(
                f"上下文长度 {start_pos + S} 超过 max_len={self.cfg.max_len} "
                f"(RoPE 位置表上限)；如需更长上下文请调大 max_len 或使用 NTK/YaRN 外推")

        cos = self.rope_cos[start_pos:start_pos + S]
        sin = self.rope_sin[start_pos:start_pos + S]
        x = self.tok_emb(idx)
        new_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_cache = block(x, cos, sin, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache:
                new_caches.append(new_cache)
        x = self.norm_f(x)

        if targets is None:
            return self.lm_head(x[:, [-1], :]), new_caches
        return None, self._chunked_cross_entropy(x, targets)

    def _chunked_cross_entropy(self, x: torch.Tensor, targets: torch.Tensor,
                               chunk_size: int = 1024) -> torch.Tensor:
        """分块计算交叉熵，避免物化完整 (B, S, V) logits（沿用旧 fused loss 的目的）。

        loss 按有效 token 数（targets != -1）归一化，语义与旧实现一致。
        """
        h = x.reshape(-1, x.size(-1))
        t = targets.reshape(-1)
        n_valid = (t != -1).sum().clamp(min=1)
        losses = []
        for i in range(0, h.size(0), chunk_size):
            logits = self.lm_head(h[i : i + chunk_size])
            # cross_entropy 在 fp32 下计算，保证混合精度训练的数值稳定性
            losses.append(F.cross_entropy(
                logits.float(), t[i : i + chunk_size], ignore_index=-1, reduction="sum"))
        return torch.stack(losses).sum() / n_valid

    def num_params(self) -> int:
        # named_parameters 默认对共享权重去重，tied embedding 只计一次
        return sum(p.numel() for p in self.parameters())


# ========== 混合精度工具 ==========
def pick_amp_dtype(device: torch.device):
    """选择混合精度 dtype：CUDA 上优先 BF16（动态范围与 FP32 一致，无需
    GradScaler），不支持的旧卡回退 FP16（需配 GradScaler），CPU 返回 None。"""
    if device.type != "cuda":
        return None
    bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    return torch.bfloat16 if bf16_ok else torch.float16


def autocast_ctx(device: torch.device, dtype):
    """CUDA + dtype 有效时启用 autocast，其余情况为空上下文。"""
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


# ========== 采样与生成 ==========
def sample_logits(logits, gen_ids=None, temperature=1.0, top_k=0, top_p=0.95,
                  repetition_penalty=1.0, seed=None):
    """temperature / top-k / top-p / repetition penalty 组合采样，返回一个 token id。

    纯 NumPy 实现，与框架无关（输入为一维 logits）。
    """
    logits = np.asarray(logits, dtype=np.float64)
    rng = np.random.RandomState(seed) if seed is not None else np.random
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    if repetition_penalty != 1.0 and gen_ids:
        for i in set(int(x) for x in gen_ids):
            if 0 <= i < len(logits):
                logits[i] = logits[i] / repetition_penalty if logits[i] >= 0 else logits[i] * repetition_penalty
    if top_k and 0 < top_k < len(logits):
        kth = np.partition(logits, -top_k)[-top_k]
        logits[logits < kth] = -np.inf
    if top_p and 0 < top_p < 1:
        order = np.argsort(logits)[::-1]
        probs = np.exp(logits[order] - logits[order].max())
        c = np.cumsum(probs)
        # 右移删除掩码：始终保留排序后的第一个 token 及首次跨过阈值的 token
        keep = np.ones(len(logits), dtype=bool)
        keep[1:] = c[:-1] <= top_p * c[-1]
        logits[order[~keep]] = -np.inf
    m = logits.max()
    if not np.isfinite(m):
        return int(rng.randint(len(logits)))
    e = np.exp(logits - m)
    p = e / e.sum()
    return int(rng.choice(len(logits), p=p))


@torch.no_grad()
def generate_ids(model: MiniGPT, prompt_ids, max_new=200, temperature=0.8,
                 top_k=50, top_p=0.95, repetition_penalty=1.0, seed=None,
                 eos_id=None, amp_dtype=None):
    """自回归生成（KV cache：预填充一次，之后每步只前向单个新 token）。

    上下文总长（prompt + 生成）达到 max_len 时停止：RoPE 位置表按 max_len
    预计算，继续生成需要 NTK/YaRN 等外推手段（本模型未启用）。
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    if seed is not None:
        np.random.seed(seed)
    prompt = [int(i) for i in prompt_ids]
    if not prompt:
        raise ValueError("prompt_ids 不能为空")
    max_len = model.cfg.max_len
    prompt = prompt[-max_len:]
    generated = list(prompt)

    # 预填充：全量前向一次，得到各层 KV cache 与末位 logits
    idx = torch.tensor([prompt], dtype=torch.long, device=device)
    with autocast_ctx(device, amp_dtype):
        logits, kv_cache = model(idx, use_cache=True)
    past = len(prompt)

    for _ in range(max_new):
        # logits: (1, 1, V)；先转 fp32 再转 numpy（numpy 不认识 bf16）
        logits_np = logits[0, -1].float().cpu().numpy()
        next_id = sample_logits(logits_np, gen_ids=generated, temperature=temperature,
                                top_k=top_k, top_p=top_p,
                                repetition_penalty=repetition_penalty)
        if eos_id is not None and next_id == eos_id:
            break
        generated.append(next_id)
        if past >= max_len:
            break
        # 增量解码：只前向新 token，复用历史 KV
        idx = torch.tensor([[next_id]], dtype=torch.long, device=device)
        with autocast_ctx(device, amp_dtype):
            logits, kv_cache = model(idx, start_pos=past, kv_cache=kv_cache,
                                     use_cache=True)
        past += 1
    if was_training:
        model.train()
    return generated
