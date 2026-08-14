# -*- coding: utf-8 -*-
"""冒烟测试：不依赖 GPU 与数据集，快速验证模型代码正确性。

用法：python test_smoke.py

覆盖：tied embeddings、RoPE 性质（恒等/保范数/相对位置）、GQA 配置、
      前向 shape、ignore_index、梯度流、因果性、KV cache 等价性、
      HF 导出映射、小批次过拟合、checkpoint 往返一致性、CPU autocast、生成接口。
（这不是训练脚本，几秒内跑完。）
"""

import os
import tempfile
from dataclasses import asdict

import torch

from model import MiniGPT, ModelConfig, apply_rope, build_rope_cache, generate_ids


def full_logits(model, idx):
    """复现 forward 的内部通路，取全部位置的 logits（供因果性/一致性断言）。"""
    B, S = idx.shape
    cos, sin = model.rope_cos[:S], model.rope_sin[:S]
    h = model.tok_emb(idx)
    for blk in model.blocks:
        h, _ = blk(h, cos, sin)
    return model.lm_head(model.norm_f(h))


def main():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_heads=4, n_kv_heads=2,
                      n_layers=2, d_ff=48, max_len=16)
    model = MiniGPT(cfg)
    n_params = model.num_params()
    print(f"[1] 构建模型 OK, 参数量: {n_params:,}")

    # tied embeddings：共享同一 Parameter，且 named_parameters 去重
    assert model.lm_head.weight is model.tok_emb.weight
    assert "lm_head.weight" not in dict(model.named_parameters())
    # 无 bias / dropout / pos_emb（LLaMA 形态）
    assert all(p.dim() != 1 or n.endswith("norm1.weight") or "norm" in n
               for n, p in model.named_parameters()), "存在意外的 1D 参数"
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert not hasattr(model, "pos_emb")
    assert "rope_cos" not in model.state_dict()  # 非持久化 buffer
    print("[2] tied embeddings / 无 bias / 无 dropout / 无 pos_emb OK")

    # RoPE 性质：位置 0 是恒等旋转；旋转保范数；内积只依赖相对距离
    cos, sin = build_rope_cache(8, 16)
    xr = torch.randn(2, 4, 16, 8)
    assert torch.allclose(apply_rope(xr, cos, sin)[:, :, :1], xr[:, :, :1]), "位置0应恒等"
    assert torch.allclose(apply_rope(xr, cos, sin).norm(dim=-1), xr.norm(dim=-1),
                          atol=1e-5), "旋转应保持范数"
    qv = torch.randn(1, 1, 1, 8).expand(1, 1, 16, 8)
    kv = torch.randn(1, 1, 1, 8).expand(1, 1, 16, 8)
    qr, kr = apply_rope(qv, cos, sin), apply_rope(kv, cos, sin)
    s = qr[0, 0] @ kr[0, 0].T  # s[i,j] = <rope(q,i), rope(k,j)>
    assert torch.allclose(s[:-1, :-1], s[1:, 1:], atol=1e-5), "内积应只依赖相对距离 i-j"
    print("[3] RoPE 性质 OK（位置0恒等 / 保范数 / 相对位置编码）")

    # GQA：MHA(n_kv=n_heads) / GQA(n_kv=2) / MQA(n_kv=1) 均可前向
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    for n_kv in (4, 2, 1):
        m = MiniGPT(ModelConfig(vocab_size=64, d_model=32, n_heads=4, n_kv_heads=n_kv,
                                n_layers=1, d_ff=48, max_len=16))
        _, l = m(x, y)
        assert torch.isfinite(l)
    # 非法配置应报错
    try:
        ModelConfig(vocab_size=64, d_model=32, n_heads=4, n_kv_heads=3)
        raise AssertionError("非法 n_kv_heads 未被拦截")
    except ValueError:
        pass
    print("[4] GQA (MHA/GQA/MQA) 与配置校验 OK")

    # 前向：训练路径返回标量 loss（logits 不物化），推理路径只返回末位 logits
    y[0, -3:] = -1  # 模拟 padding 的 ignore_index
    logits_train, loss = model(x, y)
    assert logits_train is None and loss.dim() == 0 and torch.isfinite(loss)
    logits_infer, none_loss = model(x)
    assert logits_infer.shape == (2, 1, 64) and none_loss is None
    print(f"[5] 前向 OK (loss={loss.item():.4f}, 随机初始期望≈ln(64)=4.16)")

    # 反向：所有参数都有梯度（含共享的 tok_emb.weight）
    loss.backward()
    grads = [(n, p.grad) for n, p in model.named_parameters()]
    assert all(g is not None for _, g in grads), "存在无梯度的参数"
    assert model.tok_emb.weight.grad is not None and model.tok_emb.weight.grad.abs().sum() > 0
    print(f"[6] autograd OK ({len(grads)} 个参数全部有梯度)")
    model.zero_grad(set_to_none=True)

    # 因果性：改动未来 token 不得影响过去位置的输出
    model.eval()
    with torch.no_grad():
        x2 = x.clone()
        x2[:, 8:] = (x2[:, 8:] + 1) % 64
        h1, h2 = full_logits(model, x), full_logits(model, x2)
    assert torch.allclose(h1[:, :8], h2[:, :8]), "因果 mask 失效"
    print("[7] 因果性 OK（RoPE + SDPA is_causal 生效）")

    # KV cache 等价性：预填充 + 逐 token 增量解码 与 全量重算 的 logits 一致
    with torch.no_grad():
        seq = x[0:1, :10]
        ref = full_logits(model, seq)                       # (1, 10, V) 全量参考
        lg, cache = model(seq[:, :5], use_cache=True)       # 预填充前 5 个
        assert lg.shape == (1, 1, 64)
        assert torch.allclose(lg[0, -1], ref[0, 4], atol=1e-5), "预填充 logits 不一致"
        assert len(cache) == cfg.n_layers
        assert cache[0][0].shape == (1, cfg.n_kv_heads, 5, 8)  # GQA: 缓存的是广播前 KV 头
        for i in range(5, 10):                              # 逐 token 解码
            lg, cache = model(seq[:, i:i + 1], start_pos=i,
                              kv_cache=cache, use_cache=True)
            assert torch.allclose(lg[0, -1], ref[0, i], atol=1e-5), \
                f"位置 {i} 增量解码与全量重算不一致"
    print("[8] KV cache 增量解码与全量重算一致 OK")

    # HF 导出映射：键集合完整、QKV 拆分正确、tied lm_head 不重复存储
    from export_hf import expected_hf_keys, to_hf_config, to_hf_state_dict
    hf_sd = to_hf_state_dict(model.state_dict(), cfg)
    assert set(hf_sd) == expected_hf_keys(cfg)
    hd = cfg.d_model // cfg.n_heads
    qkv = model.blocks[0].attn.qkv_proj.weight
    assert torch.equal(hf_sd["model.layers.0.self_attn.q_proj.weight"],
                       qkv[:cfg.n_heads * hd])
    assert torch.equal(hf_sd["model.layers.0.self_attn.k_proj.weight"],
                       qkv[cfg.n_heads * hd:(cfg.n_heads + cfg.n_kv_heads) * hd])
    assert torch.equal(hf_sd["model.layers.0.self_attn.v_proj.weight"],
                       qkv[(cfg.n_heads + cfg.n_kv_heads) * hd:])
    assert "lm_head.weight" not in hf_sd  # tied embeddings 只存一份
    hf_cfg = to_hf_config(cfg)
    assert (hf_cfg["hidden_size"], hf_cfg["num_key_value_heads"],
            hf_cfg["tie_word_embeddings"], hf_cfg["rope_theta"]) == (32, 2, True, 10000.0)
    print("[9] HF 导出映射（键集合/QKV 拆分/config）OK")

    # HF 数值等价：用导出后的权重按 HF Llama 的计算路径（独立实现）前向，
    # 输出必须与本模型一致 —— 验证 RoPE 约定 / QKV 拆分 / tied lm_head 全部兼容
    import torch.nn.functional as F

    def hf_style_forward(sd, cfg, idx):
        def rms(z, w):
            zf = z.float()
            return ((zf * torch.rsqrt(zf.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps))
                    * w).to(z.dtype)
        B, S = idx.shape
        hd = cfg.d_model // cfg.n_heads
        cos, sin = build_rope_cache(hd, cfg.max_len, cfg.rope_theta)
        cos, sin = cos[:S], sin[:S]
        h = F.embedding(idx, sd["model.embed_tokens.weight"])
        for i in range(cfg.n_layers):
            p = f"model.layers.{i}"
            z = rms(h, sd[f"{p}.input_layernorm.weight"])
            q = (z @ sd[f"{p}.self_attn.q_proj.weight"].T).view(B, S, cfg.n_heads, hd).transpose(1, 2)
            k = (z @ sd[f"{p}.self_attn.k_proj.weight"].T).view(B, S, cfg.n_kv_heads, hd).transpose(1, 2)
            v = (z @ sd[f"{p}.self_attn.v_proj.weight"].T).view(B, S, cfg.n_kv_heads, hd).transpose(1, 2)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            rep = cfg.n_heads // cfg.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            h = h + y.transpose(1, 2).contiguous().view(B, S, -1) @ sd[f"{p}.self_attn.o_proj.weight"].T
            z = rms(h, sd[f"{p}.post_attention_layernorm.weight"])
            h = h + (F.silu(z @ sd[f"{p}.mlp.gate_proj.weight"].T)
                     * (z @ sd[f"{p}.mlp.up_proj.weight"].T)) @ sd[f"{p}.mlp.down_proj.weight"].T
        h = rms(h, sd["model.norm.weight"])
        return h @ sd["model.embed_tokens.weight"].T  # tied lm_head

    with torch.no_grad():
        assert torch.allclose(full_logits(model, x), hf_style_forward(hf_sd, cfg, x),
                              atol=1e-5), "HF 映射权重的前向结果与本模型不一致"
    print("[10] HF 权重数值等价（独立计算路径）OK")

    # 小批次过拟合：固定 batch 训练 100 步，loss 应显著下降
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = None
    for _ in range(100):
        _, l = model(x, y)
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
        first = first if first is not None else l.item()
        last = l.item()
    print(f"[11] 过拟合 OK (loss: {first:.4f} -> {last:.4f})")
    assert last < first * 0.6, "小批次过拟合失败，训练链路可能有问题"

    # checkpoint 往返：保存/加载后输出逐位一致
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        torch.save({"model": model.state_dict(), "config": asdict(cfg)}, path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model2 = MiniGPT(ModelConfig(**ckpt["config"]))
        model2.load_state_dict(ckpt["model"])
        model2.eval()
        with torch.no_grad():
            assert torch.allclose(full_logits(model, x), full_logits(model2, x))
    print("[12] checkpoint 保存/加载往返一致 OK")

    # CPU autocast 路径（GPU 上对应 bf16/fp16，此处仅验证代码路径可执行）
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, l_bf16 = model(x, y)
    assert torch.isfinite(l_bf16)
    print("[13] autocast(bf16) 前向 OK")

    # 生成接口（KV cache 路径）
    ids = generate_ids(model, [1, 2, 3], max_new=5, temperature=1.0, seed=0)
    assert len(ids) >= 3 and ids[:3] == [1, 2, 3]
    print(f"[14] generate_ids OK ({ids})")

    print("\n全部冒烟测试通过 ✅")


if __name__ == "__main__":
    main()
