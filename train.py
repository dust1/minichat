import os
import json
import sys
import time
import math
import shutil
import numpy as np

# ========== GPU / CPU 检测 ==========
_gpu = False
try:
    import cupy as cp
    _gpu = True
    _gpu_id = int(os.environ.get("CUDA_DEVICE", "0"))
    cp.cuda.Device(_gpu_id).use()
    print(f"[GPU] CuPy detected, using GPU #{_gpu_id}")
except ImportError:
    cp = np
    _gpu = False
    print("[WARN] CuPy not installed, falling back to CPU (will be slow)")
    import subprocess
    try:
        cuda_ver = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if cuda_ver.returncode == 0:
            print(f"[INFO] NVIDIA driver: {cuda_ver.stdout.strip()}")
            print("[HINT] Install CuPy: pip install cupy-cuda12x")
    except Exception:
        pass

from dataloader import create_dataloader
import tokenizer as tok

# ========== 基础层 ==========
class Linear:
    def __init__(self, in_dim, out_dim):
        scale = cp.float32((2.0 / in_dim) ** 0.5)
        self.w = cp.random.randn(out_dim, in_dim).astype(cp.float32) * scale
        self.b = cp.zeros(out_dim, dtype=cp.float32)
        self.x = None
        self.dw = None
        self.db = None
        self.m_w = cp.zeros_like(self.w)
        self.v_w = cp.zeros_like(self.w)
        self.m_b = cp.zeros_like(self.b)
        self.v_b = cp.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.w.T + self.b

    def backward(self, grad):
        x_2d = self.x.reshape(-1, self.x.shape[-1])
        g_2d = grad.reshape(-1, grad.shape[-1])
        self.dw = g_2d.T @ x_2d
        self.db = g_2d.sum(axis=0)
        return (g_2d @ self.w).reshape(self.x.shape)

    def update(self, lr, t, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8):
        b1, b2 = betas
        self.m_w = b1 * self.m_w + (1.0 - b1) * self.dw
        self.v_w = b2 * self.v_w + (1.0 - b2) * self.dw * self.dw
        mhat = self.m_w / (1.0 - b1 ** t)
        vhat = self.v_w / (1.0 - b2 ** t)
        self.w -= lr * (mhat / (cp.sqrt(vhat) + eps) + weight_decay * self.w)

        self.m_b = b1 * self.m_b + (1.0 - b1) * self.db
        self.v_b = b2 * self.v_b + (1.0 - b2) * self.db * self.db
        mhat = self.m_b / (1.0 - b1 ** t)
        vhat = self.v_b / (1.0 - b2 ** t)
        self.b -= lr * (mhat / (cp.sqrt(vhat) + eps) + weight_decay * self.b)
        self.dw = None
        self.db = None


class Embedding:
    def __init__(self, num_emb, emb_dim):
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        scale = 1.0 / np.sqrt(emb_dim)
        self.weight = cp.random.randn(num_emb, emb_dim).astype(cp.float32) * scale
        self.x = None
        self.dw = None
        self.m_w = cp.zeros_like(self.weight)
        self.v_w = cp.zeros_like(self.weight)

    def forward(self, x):
        # x: (B, S) 整数索引
        self.x = x
        return self.weight[x]

    def backward(self, grad):
        # grad: (B, S, emb_dim)
        flat_x = self.x.reshape(-1)
        flat_grad = grad.reshape(-1, self.emb_dim)
        self.dw = cp.zeros_like(self.weight)
        cp.add.at(self.dw, flat_x, flat_grad)
        return None  # 索引不可微

    def update(self, lr, t, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8):
        b1, b2 = betas
        self.m_w = b1 * self.m_w + (1.0 - b1) * self.dw
        self.v_w = b2 * self.v_w + (1.0 - b2) * self.dw * self.dw
        mhat = self.m_w / (1.0 - b1 ** t)
        vhat = self.v_w / (1.0 - b2 ** t)
        self.weight -= lr * (mhat / (cp.sqrt(vhat) + eps) + weight_decay * self.weight)
        self.dw = None


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = cp.ones(dim, dtype=cp.float32)
        self.beta = cp.zeros(dim, dtype=cp.float32)
        self.eps = eps
        # 缓存中间变量
        self.x = None
        self.mean = None
        self.var = None
        self.x_hat = None
        self.dgamma = None
        self.dbeta = None
        self.m_gamma = cp.zeros_like(self.gamma)
        self.v_gamma = cp.zeros_like(self.gamma)
        self.m_beta = cp.zeros_like(self.beta)
        self.v_beta = cp.zeros_like(self.beta)

    def forward(self, x):
        self.x = x
        self.mean = x.mean(axis=-1, keepdims=True)
        self.var = x.var(axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / cp.sqrt(self.var + self.eps)
        out = self.gamma * self.x_hat + self.beta
        return out

    def backward(self, grad):
        # grad: (..., D)
        dout = grad
        N = self.x.shape[-1]

        dx_hat = dout * self.gamma
        dvar = cp.sum(dx_hat * (self.x - self.mean) * -0.5 * (self.var + self.eps) ** (-1.5), axis=-1, keepdims=True)
        dmean = cp.sum(dx_hat * -1.0 / cp.sqrt(self.var + self.eps), axis=-1, keepdims=True) + \
                dvar * cp.mean(-2.0 * (self.x - self.mean), axis=-1, keepdims=True)
        dx = dx_hat / cp.sqrt(self.var + self.eps) + \
             dvar * 2.0 * (self.x - self.mean) / N + \
             dmean / N

        # 对 gamma, beta 的梯度需对所有批次和序列维度求和
        self.dgamma = cp.sum(dout * self.x_hat, axis=tuple(range(grad.ndim-1)))
        self.dbeta = cp.sum(dout, axis=tuple(range(grad.ndim-1)))
        return dx

    def update(self, lr, t, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8):
        b1, b2 = betas
        self.m_gamma = b1 * self.m_gamma + (1.0 - b1) * self.dgamma
        self.v_gamma = b2 * self.v_gamma + (1.0 - b2) * self.dgamma * self.dgamma
        mhat = self.m_gamma / (1.0 - b1 ** t)
        vhat = self.v_gamma / (1.0 - b2 ** t)
        self.gamma -= lr * (mhat / (cp.sqrt(vhat) + eps) + weight_decay * self.gamma)

        self.m_beta = b1 * self.m_beta + (1.0 - b1) * self.dbeta
        self.v_beta = b2 * self.v_beta + (1.0 - b2) * self.dbeta * self.dbeta
        mhat = self.m_beta / (1.0 - b1 ** t)
        vhat = self.v_beta / (1.0 - b2 ** t)
        self.beta -= lr * (mhat / (cp.sqrt(vhat) + eps) + weight_decay * self.beta)
        self.dgamma = None
        self.dbeta = None


class Dropout:
    def __init__(self, drop_rate=0.1):
        self.drop_rate = drop_rate
        self.mask = None
        self.training = True  # 外部控制

    def forward(self, x):
        if not self.training or self.drop_rate == 0.0:
            return x
        self.mask = (cp.random.rand(*x.shape) > self.drop_rate).astype(cp.float32) / (1.0 - self.drop_rate)
        return x * self.mask

    def backward(self, grad):
        if not self.training or self.drop_rate == 0.0:
            return grad
        return grad * self.mask


# ========== 注意力与 Transformer 块 ==========
class Attention:
    def __init__(self, d_model, dropout_rate=0.1):
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.dropout = Dropout(dropout_rate)  # 用于注意力权重

    def forward(self, x, causal=False, training=True):
        self.training = training
        B, S, d = x.shape
        self.B, self.S, self.d = B, S, d
        Q = self.w_q.forward(x)
        K = self.w_k.forward(x)
        V = self.w_v.forward(x)
        scores = Q @ K.transpose(0, 2, 1) / (d ** 0.5)
        if causal:
            mask = cp.triu(cp.ones((S, S), dtype=cp.bool_), k=1)
            scores = cp.where(mask, -1e9, scores)
        e_x = cp.exp(scores - scores.max(axis=-1, keepdims=True))
        self.attn_raw = e_x / (e_x.sum(axis=-1, keepdims=True) + 1e-8)
        # 对注意力权重应用 dropout
        self.dropout.training = training
        self.attn = self.dropout.forward(self.attn_raw)
        out = self.attn @ V
        self.Q, self.K, self.V = Q, K, V
        self.x = x
        return out

    def backward(self, grad):
        # 反向传播需考虑到 dropout
        dV = self.attn.transpose(0, 2, 1) @ grad
        d_attn = grad @ self.V.transpose(0, 2, 1)
        # dropout 的反向
        d_attn = self.dropout.backward(d_attn)
        # softmax 反向
        attn = self.attn_raw
        d_scores = attn * (d_attn - (attn * d_attn).sum(axis=-1, keepdims=True))
        d_scores = d_scores / (self.d ** 0.5)
        dQ = d_scores @ self.K
        dK = d_scores.transpose(0, 2, 1) @ self.Q
        dx = self.w_q.backward(dQ) + self.w_k.backward(dK) + self.w_v.backward(dV)
        return dx

    def update(self, lr, t, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8):
        self.w_q.update(lr, t, weight_decay, betas, eps)
        self.w_k.update(lr, t, weight_decay, betas, eps)
        self.w_v.update(lr, t, weight_decay, betas, eps)


class TransformerBlock:
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        self.d_model = d_model
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.attn = Attention(d_model, dropout_rate)
        self.ffn1 = Linear(d_model, d_ff)
        self.ffn2 = Linear(d_ff, d_model)
        self.dropout1 = Dropout(dropout_rate)  # 注意力后的 dropout
        self.dropout2 = Dropout(dropout_rate)  # FFN 后的 dropout
        self.ffn1_out = None  # 用于保存 ffn1 的输出

    def forward(self, x, causal=False, training=True):
        B, S, d = x.shape
        self.B, self.S = B, S
        self.training = training

        # 注意力子层 (不变)
        normed = self.ln1.forward(x)
        attn_out = self.attn.forward(normed, causal=causal, training=training)
        self.dropout1.training = training
        x = x + self.dropout1.forward(attn_out)

        # FFN 子层 (修改处)
        normed = self.ln2.forward(x)
        h_flat = self.ffn1.forward(normed.reshape(-1, d))
        self.ffn1_out = h_flat                    # 保存 ReLU 输入
        h_flat = cp.maximum(h_flat, 0)            # ReLU
        ffn_out = self.ffn2.forward(h_flat).reshape(B, S, d)
        self.dropout2.training = training
        x = x + self.dropout2.forward(ffn_out)
        return x

    def backward(self, grad):
        B, S, d = self.B, self.S, self.d_model

        # FFN 子层反向：输入梯度 = 入口梯度 dL/dx2
        grad_ffn_out = self.dropout2.backward(grad)
        grad_ffn = self.ffn2.backward(grad_ffn_out.reshape(-1, d))
        # ReLU 梯度：使用 ffn1_out > 0
        grad_ffn = grad_ffn * (self.ffn1_out > 0)
        grad_ffn = self.ffn1.backward(grad_ffn).reshape(B, S, d)
        grad_ffn = self.ln2.backward(grad_ffn)
        grad = grad + grad_ffn   # 此刻 grad = dL/dx1（x1 = x2 的残差 + FFN 分支贡献）

        # 注意力子层反向：输入梯度 = dL/dx1（含 FFN 分支贡献）
        grad_attn_out = self.dropout1.backward(grad)
        grad_attn = self.attn.backward(grad_attn_out)
        grad_attn = self.ln1.backward(grad_attn)
        return grad + grad_attn

    def update(self, lr, t, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8):
        self.ln1.update(lr, t, weight_decay, betas, eps)
        self.ln2.update(lr, t, weight_decay, betas, eps)
        self.attn.update(lr, t, weight_decay, betas, eps)
        self.ffn1.update(lr, t, weight_decay, betas, eps)
        self.ffn2.update(lr, t, weight_decay, betas, eps)


# ========== 主模型（字节级） ==========
class ByteTransformer:
    def __init__(self, vocab_size=256, d_model=128, d_ff=256, num_blocks=4, max_len=512, dropout_rate=0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_len = max_len
        self.token_embed = Embedding(vocab_size, d_model)
        self.pos_embed = Embedding(max_len, d_model)   # 可学习位置编码
        self.dropout_emb = Dropout(dropout_rate)
        self.blocks = [TransformerBlock(d_model, d_ff, dropout_rate) for _ in range(num_blocks)]
        self.ln_f = LayerNorm(d_model)
        self.classifier = Linear(d_model, vocab_size)

    def forward(self, x, training=True):
        B, S = x.shape
        if S > self.max_len:
            S = self.max_len
            x = x[:, :S]
        pos = cp.arange(S, dtype=cp.int32)[None, :].repeat(B, axis=0)
        h = self.token_embed.forward(x) + self.pos_embed.forward(pos)
        self.dropout_emb.training = training
        h = self.dropout_emb.forward(h)
        for block in self.blocks:
            h = block.forward(h, causal=True, training=training)
        h = self.ln_f.forward(h)
        logits = self.classifier.forward(h)
        return logits

    def forward_hidden(self, x, training=True):
        """返回分类器之前的隐层状态 h (B, S, d_model)，避免物化 logits"""
        B, S = x.shape
        if S > self.max_len:
            S = self.max_len
            x = x[:, :S]
        pos = cp.arange(S, dtype=cp.int32)[None, :].repeat(B, axis=0)
        h = self.token_embed.forward(x) + self.pos_embed.forward(pos)
        self.dropout_emb.training = training
        h = self.dropout_emb.forward(h)
        for block in self.blocks:
            h = block.forward(h, causal=True, training=training)
        h = self.ln_f.forward(h)
        return h

    def backward(self, grad):
        grad = self.classifier.backward(grad)
        grad = self.ln_f.backward(grad)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        grad = self.dropout_emb.backward(grad)
        self.pos_embed.backward(grad)
        self.token_embed.backward(grad)
        return None

    def backward_from_hidden(self, dx):
        """从隐层状态 h 处开始反向传播（跳过 classifier）"""
        grad = self.ln_f.backward(dx)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        grad = self.dropout_emb.backward(grad)
        self.pos_embed.backward(grad)
        self.token_embed.backward(grad)
        return None

    def update(self, lr, t, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8):
        self.classifier.update(lr, t, weight_decay, betas, eps)
        self.ln_f.update(lr, t, weight_decay, betas, eps)
        for block in self.blocks:
            block.update(lr, t, weight_decay, betas, eps)
        self.token_embed.update(lr, t, weight_decay, betas, eps)
        self.pos_embed.update(lr, t, weight_decay, betas, eps)


# ========== 损失函数 ==========
def cross_entropy_loss(logits, targets, ignore_index=-1):
    B, S, V = logits.shape
    valid = targets != ignore_index
    n_valid = max(int(valid.sum()), 1)
    t = cp.where(valid, targets, 0)
    logits_max = logits.max(axis=-1, keepdims=True)
    exp_logits = cp.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    target_probs = cp.take_along_axis(probs, t[:, :, None], axis=-1).squeeze(-1)
    loss = -(cp.log(target_probs + 1e-8) * valid).sum() / n_valid

    grad = probs * valid[:, :, None].astype(cp.float32) / n_valid
    b_idx = cp.arange(B)[:, None]
    s_idx = cp.arange(S)[None, :]
    grad[b_idx, s_idx, t] -= valid.astype(cp.float32) / n_valid
    return loss, grad


def fused_classifier_loss(h, W, b, targets, ignore_index=-1, chunk_size=2048):
    """
    分块融合 分类器Linear + 交叉熵损失，避免物化 (B, S, V) 大张量。
    h: (B, S, d_model)  — 分类器输入
    W: (V, d_model)     — 分类器权重
    b: (V,)             — 分类器偏置
    targets: (B, S)     — 目标 token ID；为 ignore_index 的位置（padding）不参与 loss/梯度
    chunk_size: int     — 每块处理多少类
    返回 (loss, dx)，并直接设置 W/b 的梯度到传入的参数上。
    """
    B, S, d = h.shape
    V = W.shape[0]
    N = B * S

    x = h.reshape(N, d)
    t = targets.reshape(N)

    valid = t != ignore_index
    n_valid = max(int(valid.sum()), 1)
    scale = cp.float32(1.0 / n_valid)
    t_safe = cp.where(valid, t, 0)

    # === 第 1 遍：求 max_logits（数值稳定） ===
    max_logits = cp.full(N, -cp.inf, dtype=cp.float32)
    for i in range(0, V, chunk_size):
        logits = x @ W[i:i+chunk_size].T + b[i:i+chunk_size]
        cp.maximum(max_logits, logits.max(axis=1), out=max_logits)

    # === 第 2 遍：求 log-sum-exp ===
    sum_exp = cp.zeros(N, dtype=cp.float64)
    for i in range(0, V, chunk_size):
        logits = x @ W[i:i+chunk_size].T + b[i:i+chunk_size]
        sum_exp += cp.exp(logits.astype(cp.float64) - max_logits[:, None]).sum(axis=1)
    lse = max_logits + cp.log(sum_exp).astype(cp.float32)

    # 目标类别的 logit（无效位置不参与 loss）
    target_logits = (x * W[t_safe]).sum(axis=1) + b[t_safe]
    loss = -((target_logits - lse) * valid.astype(cp.float32)).sum() / n_valid

    # === 第 3 遍：求 dx, dW, db ===
    dx = cp.zeros((N, d), dtype=cp.float32)
    dW = cp.zeros_like(W)
    db = cp.zeros_like(b)
    w_valid = valid.astype(cp.float32)

    for i in range(0, V, chunk_size):
        logits = x @ W[i:i+chunk_size].T + b[i:i+chunk_size]
        probs = cp.exp(logits - lse[:, None]) * w_valid[:, None]
        dx += probs @ W[i:i+chunk_size]                          # (N, d)
        dW[i:i+chunk_size] = (probs.T @ x)                       # (chunk, d)
        db[i:i+chunk_size] = probs.sum(axis=0)                   # (chunk,)

    dx -= cp.where(valid[:, None], W[t_safe], 0)
    dx = dx.reshape(B, S, d) * scale

    t_valid = t[valid]
    if t_valid.size:
        cp.add.at(dW, t_valid, -x[valid])
        cp.add.at(db, t_valid, -1.0)
    dW *= scale
    db *= scale

    return loss, dx, dW, db


# ========== 采样 ==========
def sample_logits(logits, gen_ids=None, temperature=1.0, top_k=0, top_p=0.95,
                  repetition_penalty=1.0, seed=None):
    """temperature / top-k / top-p / repetition penalty 组合采样，返回一个 token id"""
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


# ========== 模型保存 / 加载 ==========
def _to_numpy(arr):
    return cp.asnumpy(arr) if _gpu else arr


def _rng_state_to_json(rng, cupy_style=False):
    """序列化随机数状态。

    NumPy 用 get_state()；CuPy 的 cupy.random 模块没有该接口，
    需经 get_random_state() 取 RandomState 对象再调其 get_state()。
    任一环节失败返回 None（调用方回退，不中断 checkpoint 保存）。
    """
    try:
        if cupy_style:
            state = rng.get_random_state().get_state()
        else:
            state = rng.get_state()
        return [state[0], [int(x) for x in np.asarray(state[1])],
                int(state[2]), int(state[3]), float(state[4])]
    except Exception:
        return None


def restore_rng_state(cfg):
    """从 checkpoint 配置恢复 NumPy/CuPy 随机数状态（旧 checkpoint 或序列化失败时跳过）"""
    for rng, key, cupy_style in ((np.random, "rng_np", False),
                                 (cp.random, "rng_cp", _gpu)):
        s = cfg.get(key)
        if not s:
            continue
        try:
            state = (str(s[0]), np.asarray(s[1], dtype=np.uint32),
                     int(s[2]), int(s[3]), float(s[4]))
            if cupy_style:
                rs = rng.RandomState()
                rs.set_state(state)
                rng.set_random_state(rs)
            else:
                rng.set_state(state)
        except Exception:
            pass


def save_model(model, save_dir, metadata=None, scheduler_step_count=None):
    os.makedirs(save_dir, exist_ok=True)
    # 保存所有参数及 AdamW 一阶/二阶矩
    def _save_linear(name, lin):
        np.save(os.path.join(save_dir, f"{name}_w.npy"), _to_numpy(lin.w))
        np.save(os.path.join(save_dir, f"{name}_b.npy"), _to_numpy(lin.b))
        np.save(os.path.join(save_dir, f"{name}_m_w.npy"), _to_numpy(lin.m_w))
        np.save(os.path.join(save_dir, f"{name}_v_w.npy"), _to_numpy(lin.v_w))
        np.save(os.path.join(save_dir, f"{name}_m_b.npy"), _to_numpy(lin.m_b))
        np.save(os.path.join(save_dir, f"{name}_v_b.npy"), _to_numpy(lin.v_b))
    def _save_embed(name, emb):
        np.save(os.path.join(save_dir, f"{name}_weight.npy"), _to_numpy(emb.weight))
        np.save(os.path.join(save_dir, f"{name}_m_w.npy"), _to_numpy(emb.m_w))
        np.save(os.path.join(save_dir, f"{name}_v_w.npy"), _to_numpy(emb.v_w))
    def _save_ln(name, ln):
        np.save(os.path.join(save_dir, f"{name}_gamma.npy"), _to_numpy(ln.gamma))
        np.save(os.path.join(save_dir, f"{name}_beta.npy"), _to_numpy(ln.beta))
        np.save(os.path.join(save_dir, f"{name}_m_gamma.npy"), _to_numpy(ln.m_gamma))
        np.save(os.path.join(save_dir, f"{name}_v_gamma.npy"), _to_numpy(ln.v_gamma))
        np.save(os.path.join(save_dir, f"{name}_m_beta.npy"), _to_numpy(ln.m_beta))
        np.save(os.path.join(save_dir, f"{name}_v_beta.npy"), _to_numpy(ln.v_beta))

    _save_embed("token_embed", model.token_embed)
    _save_embed("pos_embed", model.pos_embed)
    _save_ln("ln_f", model.ln_f)
    _save_linear("classifier", model.classifier)
    for i, block in enumerate(model.blocks):
        _save_ln(f"block{i}_ln1", block.ln1)
        _save_ln(f"block{i}_ln2", block.ln2)
        _save_linear(f"block{i}_attn_wq", block.attn.w_q)
        _save_linear(f"block{i}_attn_wk", block.attn.w_k)
        _save_linear(f"block{i}_attn_wv", block.attn.w_v)
        _save_linear(f"block{i}_ffn1", block.ffn1)
        _save_linear(f"block{i}_ffn2", block.ffn2)

    # 保存模型配置 + 训练元数据
    config = {
        "model_type": "ByteTransformer",
        "vocab_size": model.vocab_size,
        "d_model": model.d_model,
        "d_ff": model.d_ff,
        "num_blocks": len(model.blocks),
        "max_len": model.max_len,
        "dropout_rate": model.dropout_emb.drop_rate,
        "tokenizer_model": "bbpe.model",
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if metadata:
        config.update(metadata)
    if scheduler_step_count is not None:
        config["scheduler_step_count"] = int(scheduler_step_count)
    config["rng_np"] = _rng_state_to_json(np.random)
    if _gpu:
        config["rng_cp"] = _rng_state_to_json(cp.random, cupy_style=True)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 随 checkpoint 保存 tokenizer，保证可独立加载
    tok_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "bbpe_tokenizer")
    for fn in ("bbpe.model", "bbpe.vocab"):
        src = os.path.join(tok_dir, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(save_dir, fn))


def load_model(save_dir):
    """从 checkpoint 加载模型；优先读取 config.json，缺失时按旧格式反推架构"""
    def _load_opt_state(name, m_suffix, v_suffix, m, v):
        # 旧 checkpoint 缺少优化器状态时保持零初始化
        m_path = os.path.join(save_dir, f"{name}_{m_suffix}.npy")
        v_path = os.path.join(save_dir, f"{name}_{v_suffix}.npy")
        if os.path.isfile(m_path) and os.path.isfile(v_path):
            m[...] = cp.array(np.load(m_path))
            v[...] = cp.array(np.load(v_path))

    cfg_path = os.path.join(save_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model = ByteTransformer(
            vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
            d_ff=cfg.get("d_ff", cfg["d_model"] * 2), num_blocks=cfg["num_blocks"],
            max_len=cfg["max_len"], dropout_rate=cfg.get("dropout_rate", 0.0))
    else:
        classifier_w = np.load(os.path.join(save_dir, "classifier_w.npy"))
        num_blocks = 0
        while os.path.exists(os.path.join(save_dir, f"block{num_blocks}_ln1_gamma.npy")):
            num_blocks += 1
        ffn1_w = np.load(os.path.join(save_dir, "block0_ffn1_w.npy"))
        pos_weight = np.load(os.path.join(save_dir, "pos_embed_weight.npy"))
        model = ByteTransformer(
            vocab_size=classifier_w.shape[0], d_model=classifier_w.shape[1],
            d_ff=ffn1_w.shape[0], num_blocks=num_blocks,
            max_len=pos_weight.shape[0], dropout_rate=0.0)

    model.token_embed.weight = cp.array(np.load(os.path.join(save_dir, "token_embed_weight.npy")))
    model.pos_embed.weight = cp.array(np.load(os.path.join(save_dir, "pos_embed_weight.npy")))
    model.ln_f.gamma = cp.array(np.load(os.path.join(save_dir, "ln_f_gamma.npy")))
    model.ln_f.beta = cp.array(np.load(os.path.join(save_dir, "ln_f_beta.npy")))
    model.classifier.w = cp.array(np.load(os.path.join(save_dir, "classifier_w.npy")))
    model.classifier.b = cp.array(np.load(os.path.join(save_dir, "classifier_b.npy")))
    _load_opt_state("token_embed", "m_w", "v_w", model.token_embed.m_w, model.token_embed.v_w)
    _load_opt_state("pos_embed", "m_w", "v_w", model.pos_embed.m_w, model.pos_embed.v_w)
    _load_opt_state("ln_f", "m_gamma", "v_gamma", model.ln_f.m_gamma, model.ln_f.v_gamma)
    _load_opt_state("ln_f", "m_beta", "v_beta", model.ln_f.m_beta, model.ln_f.v_beta)
    _load_opt_state("classifier", "m_w", "v_w", model.classifier.m_w, model.classifier.v_w)
    _load_opt_state("classifier", "m_b", "v_b", model.classifier.m_b, model.classifier.v_b)
    for i in range(len(model.blocks)):
        block = model.blocks[i]
        block.ln1.gamma = cp.array(np.load(os.path.join(save_dir, f"block{i}_ln1_gamma.npy")))
        block.ln1.beta = cp.array(np.load(os.path.join(save_dir, f"block{i}_ln1_beta.npy")))
        block.ln2.gamma = cp.array(np.load(os.path.join(save_dir, f"block{i}_ln2_gamma.npy")))
        block.ln2.beta = cp.array(np.load(os.path.join(save_dir, f"block{i}_ln2_beta.npy")))
        block.attn.w_q.w = cp.array(np.load(os.path.join(save_dir, f"block{i}_attn_wq_w.npy")))
        block.attn.w_q.b = cp.array(np.load(os.path.join(save_dir, f"block{i}_attn_wq_b.npy")))
        block.attn.w_k.w = cp.array(np.load(os.path.join(save_dir, f"block{i}_attn_wk_w.npy")))
        block.attn.w_k.b = cp.array(np.load(os.path.join(save_dir, f"block{i}_attn_wk_b.npy")))
        block.attn.w_v.w = cp.array(np.load(os.path.join(save_dir, f"block{i}_attn_wv_w.npy")))
        block.attn.w_v.b = cp.array(np.load(os.path.join(save_dir, f"block{i}_attn_wv_b.npy")))
        block.ffn1.w = cp.array(np.load(os.path.join(save_dir, f"block{i}_ffn1_w.npy")))
        block.ffn1.b = cp.array(np.load(os.path.join(save_dir, f"block{i}_ffn1_b.npy")))
        block.ffn2.w = cp.array(np.load(os.path.join(save_dir, f"block{i}_ffn2_w.npy")))
        block.ffn2.b = cp.array(np.load(os.path.join(save_dir, f"block{i}_ffn2_b.npy")))
        _load_opt_state(f"block{i}_ln1", "m_gamma", "v_gamma", block.ln1.m_gamma, block.ln1.v_gamma)
        _load_opt_state(f"block{i}_ln1", "m_beta", "v_beta", block.ln1.m_beta, block.ln1.v_beta)
        _load_opt_state(f"block{i}_ln2", "m_gamma", "v_gamma", block.ln2.m_gamma, block.ln2.v_gamma)
        _load_opt_state(f"block{i}_ln2", "m_beta", "v_beta", block.ln2.m_beta, block.ln2.v_beta)
        _load_opt_state(f"block{i}_attn_wq", "m_w", "v_w", block.attn.w_q.m_w, block.attn.w_q.v_w)
        _load_opt_state(f"block{i}_attn_wq", "m_b", "v_b", block.attn.w_q.m_b, block.attn.w_q.v_b)
        _load_opt_state(f"block{i}_attn_wk", "m_w", "v_w", block.attn.w_k.m_w, block.attn.w_k.v_w)
        _load_opt_state(f"block{i}_attn_wk", "m_b", "v_b", block.attn.w_k.m_b, block.attn.w_k.v_b)
        _load_opt_state(f"block{i}_attn_wv", "m_w", "v_w", block.attn.w_v.m_w, block.attn.w_v.v_w)
        _load_opt_state(f"block{i}_attn_wv", "m_b", "v_b", block.attn.w_v.m_b, block.attn.w_v.v_b)
        _load_opt_state(f"block{i}_ffn1", "m_w", "v_w", block.ffn1.m_w, block.ffn1.v_w)
        _load_opt_state(f"block{i}_ffn1", "m_b", "v_b", block.ffn1.m_b, block.ffn1.v_b)
        _load_opt_state(f"block{i}_ffn2", "m_w", "v_w", block.ffn2.m_w, block.ffn2.v_w)
        _load_opt_state(f"block{i}_ffn2", "m_b", "v_b", block.ffn2.m_b, block.ffn2.v_b)
    return model


def _iter_params(model):
    """遍历模型全部 (参数, 梯度) 对，用于全局梯度裁剪"""
    def lin(l):
        yield l.w, l.dw
        yield l.b, l.db
    def ln(l):
        yield l.gamma, l.dgamma
        yield l.beta, l.dbeta
    for w, dw in lin(model.classifier):
        yield w, dw
    for g, dg in ln(model.ln_f):
        yield g, dg
    for blk in model.blocks:
        for g, dg in ln(blk.ln1):
            yield g, dg
        for g, dg in ln(blk.ln2):
            yield g, dg
        for layer in (blk.attn.w_q, blk.attn.w_k, blk.attn.w_v, blk.ffn1, blk.ffn2):
            for w, dw in lin(layer):
                yield w, dw
    yield model.token_embed.weight, model.token_embed.dw
    yield model.pos_embed.weight, model.pos_embed.dw


def clip_gradients(model, max_norm=1.0):
    """全局梯度范数裁剪（替代原来的逐层裁剪）"""
    sq = 0.0
    for _, g in _iter_params(model):
        if g is not None:
            sq += float(_to_numpy((g * g).sum()))
    norm = sq ** 0.5
    if norm > max_norm and norm > 0:
        scale = max_norm / norm
        for _, g in _iter_params(model):
            if g is not None:
                g *= scale
    return norm


def evaluate(model, val_ds, batch_size=32, ignore_index=-1):
    """验证集 loss（training=False），用于报告 loss / perplexity"""
    total, n = 0.0, 0
    for start in range(0, len(val_ds), batch_size):
        idx = list(range(start, min(start + batch_size, len(val_ds))))
        x, y = val_ds.get_batch(idx)
        y_cp = cp.array(y)
        h = model.forward_hidden(cp.array(x), training=False)
        loss, _, _, _ = fused_classifier_loss(
            h, model.classifier.w, model.classifier.b, y_cp, ignore_index=ignore_index)
        n_valid = max(int((y_cp != ignore_index).sum()), 1)
        total += float(loss) * n_valid
        n += n_valid
    return total / max(n, 1)


# ========== 学习率调度器 ==========
class LRScheduler:
    def __init__(self, optimizer_lr, warmup_steps, total_steps, min_lr=1e-5):
        self.lr = optimizer_lr
        self.base_lr = optimizer_lr
        self.warmup_steps = warmup_steps
        self.total_steps = max(1, total_steps)
        self.min_lr = min_lr
        self.step_count = 0

    def get_lr(self):
        if self.step_count < self.warmup_steps:
            # 线性 warmup
            lr = self.base_lr * (self.step_count + 1) / self.warmup_steps
        else:
            # 余弦退火
            progress = min(1.0, max(0.0, (self.step_count - self.warmup_steps) /
                                    max(1, self.total_steps - self.warmup_steps)))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        self.step_count += 1
        return lr


# ========== 主训练流程 ==========
def main():
    device = "GPU" if _gpu else "CPU"
    print(f"[INFO] Training on {device}")
    DATA_DIR = r"./out/tokenized_data"
    SAVE_DIR = r"./model_weights"
    MAX_LEN = 512
    STRIDE = 256
    BATCH_SIZE = 64
    EPOCHS = 30
    BASE_LR = 3e-4
    GRAD_CLIP = 1.0
    WARMUP_STEPS = 200
    D_MODEL = 128
    D_FF = 256
    NUM_BLOCKS = 4
    DROPOUT = 0.1
    WEIGHT_DECAY = 0.1
    BETAS = (0.9, 0.999)
    EPS = 1e-8
    VAL_BATCH = 32
    SEED = 42
    RESUME = "--resume" in sys.argv

    cp.random.seed(SEED)
    np.random.seed(SEED)

    print("Loading tokenized dataset...")
    train_ds = create_dataloader(DATA_DIR, split="train", max_len=MAX_LEN, stride=STRIDE)
    val_ds = create_dataloader(DATA_DIR, split="val", max_len=MAX_LEN, stride=STRIDE)
    n_samples = len(train_ds)
    if n_samples == 0:
        print("ERROR: No training samples. Run tokenizer.py first.")
        return
    print(f"Train samples: {n_samples}, Val samples: {len(val_ds)}, vocab_size: {train_ds.vocab_size}")

    resume_cfg = None
    if RESUME and os.path.isdir(SAVE_DIR) and os.path.isfile(os.path.join(SAVE_DIR, "config.json")):
        print("Resuming from checkpoint...")
        model = load_model(SAVE_DIR)
        with open(os.path.join(SAVE_DIR, "config.json"), "r", encoding="utf-8") as f:
            resume_cfg = json.load(f)
        restore_rng_state(resume_cfg)
    else:
        model = ByteTransformer(
            vocab_size=train_ds.vocab_size, d_model=D_MODEL, d_ff=D_FF,
            num_blocks=NUM_BLOCKS, max_len=MAX_LEN, dropout_rate=DROPOUT
        )

    # 小数据量防御：total_steps 至少为 1，warmup 不超过总步数一半
    # 每轮实际批次数为 ceil(n_samples / BATCH_SIZE)（含最后一个不完整 batch）
    total_steps = EPOCHS * max(1, math.ceil(n_samples / BATCH_SIZE))
    warmup_steps = max(1, min(WARMUP_STEPS, total_steps // 2))
    scheduler = LRScheduler(BASE_LR, warmup_steps, total_steps)
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    best_val_loss = float("inf")
    opt_step = 0
    start_epoch = 0
    if resume_cfg:
        opt_step = int(resume_cfg.get("opt_step", 0))
        best_val_loss = float(resume_cfg.get("best_val_loss", best_val_loss))
        start_epoch = int(resume_cfg.get("epoch", 0))
        scheduler.step_count = int(resume_cfg.get(
            "scheduler_step_count", resume_cfg.get("opt_step", 0)))
        print(f"Resumed at epoch {start_epoch}, opt_step {opt_step}, "
              f"scheduler step {scheduler.step_count}, best_val_loss {best_val_loss}")

    for epoch in range(start_epoch, EPOCHS):
        perm = cp.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, BATCH_SIZE):
            idx = perm[start:start+BATCH_SIZE].tolist()
            x_batch_np, y_batch_np = train_ds.get_batch(idx)

            if _gpu:
                x_batch = cp.array(x_batch_np)
                y_batch = cp.array(y_batch_np)
            else:
                x_batch = x_batch_np
                y_batch = y_batch_np

            h = model.forward_hidden(x_batch, training=True)
            loss, dx, dW, db = fused_classifier_loss(
                h, model.classifier.w, model.classifier.b, y_batch, ignore_index=-1)
            model.classifier.dw = dW
            model.classifier.db = db
            model.backward_from_hidden(dx)

            clip_gradients(model, GRAD_CLIP)

            lr = scheduler.get_lr()
            opt_step += 1
            model.update(lr, opt_step, WEIGHT_DECAY, BETAS, EPS)

            epoch_loss += float(loss)
            n_batches += 1

            if n_batches % 50 == 0:
                print(f"  Batch {n_batches:4d}, Loss: {loss:.6f}, LR: {lr:.2e}")

        avg_loss = epoch_loss / n_batches
        line = f"Epoch {epoch+1:3d}/{EPOCHS}, Avg Train Loss: {avg_loss:.6f}"

        if len(val_ds) > 0:
            val_loss = evaluate(model, val_ds, batch_size=VAL_BATCH)
            ppl = float(np.exp(val_loss))
            line += f", Val Loss: {val_loss:.6f}, PPL: {ppl:.3f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, SAVE_DIR, metadata={
                    "epoch": epoch + 1, "opt_step": opt_step,
                    "best_val_loss": float(best_val_loss), "train_loss": avg_loss, "lr": lr,
                }, scheduler_step_count=scheduler.step_count)
                line += "  [saved]"
        print(line)

    save_model(model, SAVE_DIR, metadata={
        "epoch": EPOCHS, "opt_step": opt_step,
        "best_val_loss": float(best_val_loss), "final_train_loss": avg_loss,
    }, scheduler_step_count=scheduler.step_count)

    # ===== 生成测试 =====
    print("\n--- Generating sample ---")
    prompt_ids = train_ds.prompt_ids(30)
    prompt_text = tok.decode(prompt_ids)
    print(f"Prompt: {prompt_text}")

    generated = list(prompt_ids)
    for _ in range(200):
        inp = cp.array([generated[-MAX_LEN:]], dtype=cp.int32)
        logits = model.forward(inp, training=False)
        logits_np = _to_numpy(logits[0, -1, :])
        next_id = sample_logits(logits_np, gen_ids=generated, temperature=0.8,
                                top_k=50, top_p=0.95, repetition_penalty=1.05)
        if next_id == tok.EOS_ID:
            break
        generated.append(next_id)
        if len(generated) > MAX_LEN:
            generated = generated[-MAX_LEN:]

    gen_text = tok.decode(generated)
    print(f"Generated: {gen_text[:300]}")


if __name__ == "__main__":
    main()