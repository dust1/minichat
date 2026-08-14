# -*- coding: utf-8 -*-
"""MiniChat PyTorch 训练器（第一批现代化改造）。

相对旧 CuPy 手写版本的变化：
  - 反向传播全部交由 PyTorch autograd（删除手写 backward 与逐层 AdamW）
  - 标准 LLaMA 形态架构：RoPE + GQA + RMSNorm + SwiGLU + tied embeddings，
    无 bias / dropout（见 model.py）
  - BF16（优先）/ FP16 混合精度训练
  - AdamW 参数分组：norm 等 1D 参数不做 weight decay
  - TensorBoard 日志：loss / lr / 梯度范数 / 吞吐 / 验证 PPL
  - checkpoint：ckpt.pt（验证最优）与 ckpt_last.pt（最终状态），不再互相覆盖

用法：
    python train.py            # 从头训练
    python train.py --resume   # 从 checkpoint 恢复
查看训练曲线：
    tensorboard --logdir out/runs
"""

import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

from dataloader import create_dataloader
import tokenizer as tok
from model import MiniGPT, ModelConfig, autocast_ctx, generate_ids, pick_amp_dtype

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


# ========== 优化器与学习率 ==========
def configure_optimizer(model, lr, weight_decay, betas, eps, device):
    """AdamW 参数分组：>=2D 的权重矩阵做 weight decay，1D（norm 增益等）不做。"""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # CUDA 上优先 fused AdamW（一次 kernel 完成更新）；不支持时回退普通实现
    try:
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps,
                                 fused=(device.type == "cuda"))
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


def lr_at(step, base_lr, warmup_steps, total_steps, min_lr):
    """线性 warmup -> 余弦退火 -> min_lr（无状态，由 step 直接推导，便于恢复）。"""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min(1.0, max(0.0, (step - warmup_steps) /
                                max(1, total_steps - warmup_steps)))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ========== Checkpoint ==========
def save_checkpoint(model, optimizer, save_dir, *, step, epoch, best_val_loss,
                    train_loss=None, lr=None, filename="ckpt.pt"):
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(model.cfg),
        "step": int(step),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_np": np.random.get_state(),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(payload, os.path.join(save_dir, filename))

    # 人类可读配置（仅供查看；加载走 ckpt 内的 config 字段）
    cfg = asdict(model.cfg)
    cfg.update({
        "model_type": "MiniGPT",
        "step": int(step),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "tokenizer_model": "bbpe.model",
        "saved_at": payload["saved_at"],
    })
    if train_loss is not None:
        cfg["train_loss"] = float(train_loss)
    if lr is not None:
        cfg["lr"] = float(lr)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # 随 checkpoint 保存 tokenizer，保证可独立加载
    tok_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "out", "bbpe_tokenizer")
    for fn in ("bbpe.model", "bbpe.vocab"):
        src = os.path.join(tok_dir, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(save_dir, fn))


def load_checkpoint(save_dir, device):
    """优先恢复 ckpt_last.pt（训练结束时的完整状态），其次 ckpt.pt（验证最优）。"""
    for name in ("ckpt_last.pt", "ckpt.pt"):
        path = os.path.join(save_dir, name)
        if os.path.isfile(path):
            ckpt = torch.load(path, map_location=device, weights_only=False)
            model = MiniGPT(ModelConfig(**ckpt["config"])).to(device)
            model.load_state_dict(ckpt["model"])
            return model, ckpt
    raise FileNotFoundError(f"在 {save_dir} 中找不到 ckpt.pt / ckpt_last.pt")


def restore_rng(ckpt):
    """恢复 NumPy / PyTorch 随机数状态（失败时跳过，不影响权重）。"""
    try:
        if ckpt.get("rng_torch") is not None:
            torch.set_rng_state(ckpt["rng_torch"])
        if ckpt.get("rng_cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        if ckpt.get("rng_np") is not None:
            np.random.set_state(ckpt["rng_np"])
    except Exception as e:
        print(f"[WARN] RNG 状态恢复失败（忽略）: {e}")


# ========== 评估 ==========
@torch.no_grad()
def evaluate(model, val_ds, device, amp_dtype, batch_size=32):
    """验证集 loss，按有效 token 数加权（与训练 loss 归一化口径一致）。"""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for start in range(0, len(val_ds), batch_size):
        idx = list(range(start, min(start + batch_size, len(val_ds))))
        x_np, y_np = val_ds.get_batch(idx)
        x = torch.from_numpy(x_np).long().to(device)
        y = torch.from_numpy(y_np).long().to(device)
        with autocast_ctx(device, amp_dtype):
            _, loss = model(x, y)
        n_valid = int((y != -1).sum().item())
        total += float(loss) * n_valid
        n += n_valid
    if was_training:
        model.train()
    return total / max(n, 1)


# ========== 主训练流程 ==========
def main():
    DATA_DIR = r"./out/tokenized_data"
    SAVE_DIR = r"./model_weights"
    MAX_LEN = 512
    STRIDE = 256
    BATCH_SIZE = 64
    EPOCHS = 30
    BASE_LR = 3e-4
    MIN_LR = 1e-5
    GRAD_CLIP = 1.0
    WARMUP_STEPS = 200
    D_MODEL = 128
    N_HEADS = 4
    N_KV_HEADS = 2      # GQA：KV 头数；与 N_HEADS 相等时退化为 MHA
    N_LAYERS = 4
    D_FF = 176          # SwiGLU 三矩阵下与旧 d_ff=256 ReLU FFN 参数量相当
    ROPE_THETA = 10000.0
    WEIGHT_DECAY = 0.1
    BETAS = (0.9, 0.999)
    EPS = 1e-8
    VAL_BATCH = 32
    SEED = 42
    LOG_INTERVAL = 50
    RESUME = "--resume" in sys.argv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = pick_amp_dtype(device)
    # GradScaler 只在 FP16 下需要；BF16/FP32 下为无操作
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"[INFO] device={device.type}, "
          f"autocast={amp_dtype if amp_dtype is not None else 'off(fp32)'}")

    print("Loading tokenized dataset...")
    train_ds = create_dataloader(DATA_DIR, split="train", max_len=MAX_LEN, stride=STRIDE)
    val_ds = create_dataloader(DATA_DIR, split="val", max_len=MAX_LEN, stride=STRIDE)
    n_samples = len(train_ds)
    if n_samples == 0:
        print("ERROR: No training samples. Run tokenizer.py first.")
        return
    print(f"Train samples: {n_samples}, Val samples: {len(val_ds)}, "
          f"vocab_size: {train_ds.vocab_size}")

    resume_ckpt = None
    if RESUME and os.path.isdir(SAVE_DIR):
        try:
            model, resume_ckpt = load_checkpoint(SAVE_DIR, device)
            print("Resuming from checkpoint...")
        except FileNotFoundError:
            model = None
    else:
        model = None

    if model is None:
        model = MiniGPT(ModelConfig(
            vocab_size=train_ds.vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
            n_kv_heads=N_KV_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
            max_len=MAX_LEN, rope_theta=ROPE_THETA,
        )).to(device)

    n_params = model.num_params()
    print(f"[INFO] params: {n_params/1e6:.2f}M ({n_params:,})")
    print(f"[INFO] arch(LLaMA-style): d_model={model.cfg.d_model}, "
          f"n_heads={model.cfg.n_heads}, n_kv_heads={model.cfg.n_kv_heads}, "
          f"n_layers={model.cfg.n_layers}, d_ff={model.cfg.d_ff}(SwiGLU), "
          f"rope_theta={model.cfg.rope_theta}, max_len={model.cfg.max_len}, "
          f"tied_embeddings=True")

    optimizer = configure_optimizer(model, BASE_LR, WEIGHT_DECAY, BETAS, EPS, device)

    # 小数据量防御：total_steps 至少为 1，warmup 不超过总步数一半
    # 每轮实际批次数为 ceil(n_samples / BATCH_SIZE)（含最后一个不完整 batch）
    total_steps = EPOCHS * max(1, math.ceil(n_samples / BATCH_SIZE))
    warmup_steps = max(1, min(WARMUP_STEPS, total_steps // 2))
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    best_val_loss = float("inf")
    step = 0
    start_epoch = 0
    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        restore_rng(resume_ckpt)
        step = int(resume_ckpt.get("step", 0))
        best_val_loss = float(resume_ckpt.get("best_val_loss", best_val_loss))
        start_epoch = int(resume_ckpt.get("epoch", 0))
        print(f"Resumed at epoch {start_epoch}, step {step}, "
              f"best_val_loss {best_val_loss}")

    writer = None
    if SummaryWriter is not None:
        run_dir = os.path.join("out", "runs", time.strftime("%Y%m%d_%H%M%S"))
        writer = SummaryWriter(run_dir)
        writer.add_text("config", json.dumps({
            **asdict(model.cfg), "batch_size": BATCH_SIZE, "epochs": EPOCHS,
            "base_lr": BASE_LR, "min_lr": MIN_LR, "warmup_steps": warmup_steps,
            "total_steps": total_steps, "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP, "amp_dtype": str(amp_dtype), "seed": SEED,
        }, ensure_ascii=False, indent=2))
        print(f"[INFO] TensorBoard logdir: {run_dir}  (tensorboard --logdir out/runs)")
    else:
        print("[WARN] 未安装 tensorboard，只输出控制台日志 (pip install tensorboard)")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        win_loss, win_tokens = 0.0, 0
        t0 = time.time()

        for start in range(0, n_samples, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            x_np, y_np = train_ds.get_batch([int(i) for i in idx])
            x = torch.from_numpy(x_np).long().to(device)
            y = torch.from_numpy(y_np).long().to(device)

            lr = lr_at(step, BASE_LR, warmup_steps, total_steps, MIN_LR)
            for g in optimizer.param_groups:
                g["lr"] = lr

            with autocast_ctx(device, amp_dtype):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            n_batches += 1
            loss_f = float(loss)
            epoch_loss += loss_f
            win_loss += loss_f
            win_tokens += x.numel()

            if n_batches % LOG_INTERVAL == 0:
                dt = time.time() - t0
                tps = win_tokens / max(dt, 1e-9)
                print(f"  Batch {n_batches:4d}, Loss: {win_loss/LOG_INTERVAL:.6f}, "
                      f"LR: {lr:.2e}, GradNorm: {float(grad_norm):.3f}, "
                      f"{tps:,.0f} tok/s")
                if writer is not None:
                    writer.add_scalar("train/loss", win_loss / LOG_INTERVAL, step)
                    writer.add_scalar("train/lr", lr, step)
                    writer.add_scalar("train/grad_norm", float(grad_norm), step)
                    writer.add_scalar("train/tokens_per_sec", tps, step)
                win_loss, win_tokens = 0.0, 0
                t0 = time.time()

        avg_loss = epoch_loss / n_batches
        line = f"Epoch {epoch+1:3d}/{EPOCHS}, Avg Train Loss: {avg_loss:.6f}"
        if writer is not None:
            writer.add_scalar("epoch/train_loss", avg_loss, epoch + 1)

        if len(val_ds) > 0:
            val_loss = evaluate(model, val_ds, device, amp_dtype, batch_size=VAL_BATCH)
            ppl = float(np.exp(min(val_loss, 20.0)))
            line += f", Val Loss: {val_loss:.6f}, PPL: {ppl:.3f}"
            if writer is not None:
                writer.add_scalar("val/loss", val_loss, step)
                writer.add_scalar("val/ppl", ppl, step)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, SAVE_DIR, step=step, epoch=epoch + 1,
                                best_val_loss=best_val_loss, train_loss=avg_loss,
                                lr=lr, filename="ckpt.pt")
                line += "  [saved best]"
        print(line)

    # 最终状态单独保存，不覆盖验证最优的 ckpt.pt
    save_checkpoint(model, optimizer, SAVE_DIR, step=step, epoch=EPOCHS,
                    best_val_loss=best_val_loss, train_loss=avg_loss,
                    filename="ckpt_last.pt")
    if len(val_ds) == 0:
        # 无验证集时 ckpt.pt 从未写过，补一份方便推理直接加载
        save_checkpoint(model, optimizer, SAVE_DIR, step=step, epoch=EPOCHS,
                        best_val_loss=best_val_loss, train_loss=avg_loss,
                        filename="ckpt.pt")

    # ===== 生成测试 =====
    print("\n--- Generating sample ---")
    prompt_ids = train_ds.prompt_ids(30)
    print(f"Prompt: {tok.decode(prompt_ids)}")
    gen_ids = generate_ids(model, prompt_ids, max_new=200, temperature=0.8,
                           top_k=50, top_p=0.95, repetition_penalty=1.05,
                           eos_id=tok.EOS_ID, amp_dtype=amp_dtype)
    print(f"Generated: {tok.decode(gen_ids)[:300]}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
