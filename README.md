# MiniChat：Decoder-only Transformer 学习项目

MiniChat 是一个小型自回归语言模型学习项目，覆盖从 tokenizer 训练、数据切分、next-token 训练、验证到文本生成的完整链路。

当前版本已完成三批现代化改造：**第一批**（PyTorch 迁移：autograd / SDPA / RMSNorm / SwiGLU / tied embeddings / BF16 / TensorBoard）、**第二批**（标准 LLaMA 形态：RoPE + GQA + 无 bias/dropout）、**第三批**（KV cache 增量推理 + safetensors/HF 导出 + vLLM/SGLang 对接）。

> 受资源限制，在20MB的中文语料上完成过训练，能够正常推理出东西，但是属于前言不搭后语的程度，不过至少能说话不会乱码

## 分支记录
learn_1 - 最初版本，使用Numpy/cupy手动实现的小写GPT类模型

## 当前完成状态

第一批（PyTorch 迁移）、第二批（LLaMA 形态架构对齐）与第三批（推理与部署）的功能已经实现：

| 修改点 | 实现状态 | 说明 |
|---|---|---|
| PyTorch autograd | 已实现 | 删除全部手写 backward / 逐层 AdamW，反向传播由 autograd 完成 |
| GQA + SDPA | 已实现 | 4 Q 头 / 2 KV 头分组查询注意力，融合 QKV 投影 + 输出投影，`F.scaled_dot_product_attention` 自动选择高速内核，注意力显存 O(S²)→O(S) |
| RoPE | 已实现 | 旋转位置编码（θ=10000），替代可学习绝对位置 embedding，位置信息只注入 Q/K |
| RMSNorm | 已实现 | 替代 LayerNorm，fp32 下计算均方根保证混合精度稳定性 |
| SwiGLU | 已实现 | `down(silu(gate(x)) * up(x))` 三矩阵 FFN，`d_ff=176` 与旧 256 ReLU FFN 参数量相当 |
| Tied embeddings | 已实现 | 输出头与输入 embedding 共享权重，小模型参数省约 35% |
| 无 bias / dropout | 已实现 | 全部 Linear 无 bias，模型无 dropout 层（现代 LLM 惯例，同时保证 SDPA 走高速内核） |
| KV cache 增量推理 | 已实现 | 预填充一次后逐 token 解码，每步只前向新 token；GQA 缓存广播前的 KV 头，cache 显存减半 |
| safetensors / HF 导出 | 已实现 | `export_hf.py` 导出 LlamaForCausalLM 兼容目录（config.json + model.safetensors + tokenizer） |
| vLLM / SGLang 对接 | 已实现 | 导出目录可直接 `vllm serve`（需 Linux 环境） |
| BF16/FP16 混合精度 | 已实现 | `torch.autocast`，优先 BF16（无需 GradScaler），旧卡回退 FP16 + GradScaler |
| TensorBoard 日志 | 已实现 | loss / lr / 梯度范数 / tokens/s / 验证 PPL，日志写入 `out/runs/` |
| 训练与推理统一 tokenizer | 已实现 | 均使用 SentencePiece BBPE；推理从 checkpoint 加载 tokenizer |
| EOS、文档边界、padding 与 mask | 已实现 | 窗口不跨文档，padding target 使用 `ignore_index=-1` |
| 验证集、验证 loss 与 perplexity | 已实现 | 验证 loss 按有效 token 数加权 |
| AdamW 参数分组与全局梯度裁剪 | 已实现 | 1D 参数（norm 增益）不做 weight decay |
| Checkpoint 保存与恢复 | 已实现 | `ckpt.pt`（验证最优）与 `ckpt_last.pt`（最终状态）分离，不再互相覆盖 |
| 可控采样 | 已实现 | 支持 temperature、top-k、top-p、重复惩罚和 seed |
| 冒烟测试 | 已实现 | `test_smoke.py`：RoPE 性质、GQA、KV cache 等价性、HF 映射、因果性、梯度流、过拟合、checkpoint 往返 |

后续批次计划：gradient checkpointing、FSDP 多卡训练。

## 技术架构

```text
UTF-8 文本文档
      │
      ▼
SentencePiece BBPE tokenizer
      │  文档末尾追加 EOS
      ▼
train_data.bin / val_data.bin
      │  文档内滑动窗口 + padding + ignore mask
      ▼
Token Embedding（位置信息由 RoPE 在注意力内注入）
      │
      ▼
4 × Pre-Norm Transformer Block
      │
      ├─ RMSNorm → GQA Causal Self-Attention (RoPE + SDPA) → Residual
      │
      └─ RMSNorm → SwiGLU FFN → Residual
      │
      ▼
Final RMSNorm
      │
      ▼
LM Head（与 Token Embedding 共享权重）
      │
      ▼
Next-token Cross Entropy（分块计算，不物化完整 logits）
```

### 默认模型配置

默认超参数定义在 `train.py`：

| 参数 | 默认值 |
|---|---:|
| 架构 | Decoder-only Transformer（PyTorch，LLaMA 形态） |
| Transformer Block | 4 |
| 隐藏维度 `d_model` | 128 |
| 注意力 | GQA：4 Q 头 / 2 KV 头（SDPA，head_dim=32） |
| 位置编码 | RoPE（θ=10000） |
| FFN | SwiGLU，`d_ff` = 176 |
| 归一化 | RMSNorm（eps=1e-6） |
| bias / Dropout | 无 / 无 |
| 最大上下文 | 512 tokens |
| Batch size | 64 |
| Epochs | 30 |
| 基础学习率 | 3e-4 |
| Warmup | 最多 200 steps，且不超过总步数一半 |
| 最低学习率 | 1e-5 |
| 优化器 | `torch.optim.AdamW`（CUDA 上 fused） |
| Weight decay | 0.1（仅 ≥2D 权重矩阵） |
| 全局梯度裁剪 | 1.0 |
| 混合精度 | BF16 优先，旧卡回退 FP16 + GradScaler，CPU 关闭 |
| 数据窗口 stride | 256 |

词表大小由语料规模动态决定，范围为 2,000～32,000。默认架构（GQA n_kv_heads=2）的参数量约为：

```text
参数量 = 468,096 + 128 × vocab_size
```

例如词表为 8,000 时，模型约有 149 万参数（tied embeddings 下 embedding 只计一份）。

## 核心模块设计

### 1. Tokenizer

`token_train.py` 使用 SentencePiece 训练 BPE tokenizer：

- `model_type=bpe`
- `byte_fallback=True`，未覆盖字符可退化到 UTF-8 字节
- identity normalization，尽量保留原始文本
- 固定特殊 token：
  - `UNK=0`
  - `BOS=1`
  - `EOS=2`
  - `PAD=3`
- 不使用 dummy prefix，不主动压缩连续空白

`tokenizer.py` 负责：

- 使用同一 SentencePiece 模型编码原始 Unicode 文本；
- 在每篇文档末尾加入 EOS；
- 按文档划分训练集和验证集；
- 生成 `int32` 二进制 token 文件和 `meta.json`；
- 从 checkpoint 目录加载 tokenizer，避免推理词表错配。

### 2. 数据加载

`dataloader.py` 以 EOS 为文档边界构建滑动窗口：

- 窗口不会跨越文档；
- 短文档和尾部窗口使用 PAD 补齐输入；
- 无效 target 使用 `ignore_index=-1`；
- 窗口同时记录 `window_end` 和 `doc_end`，保证完整窗口不会丢失最后一个 next-token 目标；
- target 是 input 向右平移一位的序列。

示例：

```text
x = [20, 21, 22, 23]
y = [21, 22, 23, 24]
```

文档尾部：

```text
x = [24, 25, 26, EOS]
y = [25, 26, EOS, IGNORE]
```

### 3. Transformer（model.py）

模型为标准 LLaMA 形态的 Pre-Norm 残差结构。每个 Block 包含：

1. RMSNorm；
2. GQA 因果自注意力：融合 QKV 投影（Q 4 头，K/V 各 2 头）→ 按头拆分 → Q/K 施加 RoPE → KV 头广播至 Q 头数 → `F.scaled_dot_product_attention(is_causal=True)` → 输出投影；
3. 残差连接；
4. RMSNorm；
5. SwiGLU FFN：`down(silu(gate(x)) * up(x))`；
6. 残差连接。

RoPE（旋转位置编码）以非持久化 buffer 形式预计算 cos/sin 缓存（按 `max_len` 与 θ=10000），只作用于 Q/K；位置 i 的 query 与位置 j 的 key 的内积只依赖相对距离 (i-j)，因此具备长度外推的基础。GQA 将 KV cache 需求降为 MHA 的 n_kv_heads/n_heads（默认 1/2），为后续推理加速预留结构。

所有 Linear 均无 bias，模型无 dropout。注意力计算由 SDPA 内核完成，不物化 S×S 注意力矩阵，显存从 O(S²) 降为 O(S)（flash/mem-efficient 后端，需 GPU + bf16/fp16）。

反向传播由 PyTorch autograd 自动完成。权重初始化为 GPT-2 惯例：`normal(0, 0.02)`，残差分支输出投影（`out_proj`/`down_proj`）按 `0.02/√(2·n_layers)` 缩小初始化。

### 4. 损失函数

训练目标为标准 next-token cross entropy，`ignore_index=-1` 的 padding target 不参与 loss 或梯度，loss 按有效 token 数归一化。

为控制峰值显存，loss 在序列维上**分块计算**（每块 1024 token 的 logits），不物化完整 `(batch, sequence, vocabulary)` logits——与旧版手写 `fused_classifier_loss` 目的一致，但梯度由 autograd 处理。推理时只计算最后一个位置的 logits。

### 5. 优化器、学习率与混合精度

优化器为 `torch.optim.AdamW`（CUDA 上使用 fused 实现），参数分两组：

- ≥2D 的权重矩阵：`weight_decay=0.1`；
- 1D 参数（RMSNorm 增益）：不做 weight decay。

所有梯度统一计算全局 L2 norm，超过 1.0 按同一比例缩放；梯度范数同时写入 TensorBoard，可用于诊断 loss spike。

学习率策略（无状态函数，由 step 直接推导，便于恢复）：

```text
线性 warmup → cosine decay → min_lr
```

混合精度：CUDA 上优先 BF16（动态范围与 FP32 一致，无需 GradScaler），不支持 BF16 的旧卡回退 FP16 + GradScaler，CPU 训练保持 FP32。RMSNorm 与 cross entropy 内部在 fp32 下计算，保证数值稳定。

### 6. Checkpoint

`model_weights/` 中保存：

- `ckpt.pt`：验证 loss 最优时的完整状态；
- `ckpt_last.pt`：训练结束时的完整状态（供 `--resume` 优先恢复）；
- 两者均包含：模型 `state_dict`、AdamW 状态、模型配置、步数/epoch/best_val_loss、NumPy/PyTorch RNG 状态；
- `config.json`：人类可读的模型配置与训练进度（不用于加载）；
- `bbpe.model` 和 `bbpe.vocab`。

`python train.py --resume` 可从 checkpoint 恢复权重、优化器状态和学习率进度。

注意：checkpoint 格式与旧 CuPy 时代的散装 `.npy`、以及第一批（RoPE/GQA 之前）的 `ckpt.pt` 均不兼容。

### 7. 训练日志

训练指标写入 `out/runs/<timestamp>/`：

```bash
tensorboard --logdir out/runs
```

记录内容：`train/loss`、`train/lr`、`train/grad_norm`、`train/tokens_per_sec`（每 50 step）、`val/loss`、`val/ppl`、`epoch/train_loss`（每 epoch）以及完整超参数文本。未安装 tensorboard 时自动退化为仅控制台输出。

## 项目结构

```text
minichat/
├── train_dataset/       # 原始 UTF-8 .txt 训练语料
├── out/
│   ├── bbpe_tokenizer/  # SentencePiece 模型
│   ├── tokenized_data/  # train/val 二进制 token 数据
│   └── runs/            # TensorBoard 日志
├── model_weights/       # checkpoint、配置和 tokenizer
├── logs/                # start.sh 运行后生成的阶段日志
├── token_train.py       # 训练 BBPE tokenizer
├── tokenizer.py         # 数据 token 化与 train/val 划分
├── dataloader.py        # 文档窗口和 batch 构造
├── model.py             # PyTorch 模型定义（RoPE/GQA/RMSNorm/SwiGLU/tied embeddings/KV cache）
├── train.py             # 训练循环、混合精度、日志与 checkpoint
├── infer.py             # 文本生成入口（KV cache 增量解码）
├── export_hf.py         # 导出 HF/safetensors 格式（对接 vLLM/SGLang）
├── test_smoke.py        # CPU 冒烟测试（不依赖 GPU 与数据集）
├── check_deps.py        # 环境依赖检查
└── start.sh             # Linux 后台一键训练脚本
```

## 环境准备

依赖：

```bash
python3 -m pip install numpy sentencepiece torch tensorboard
```

使用 NVIDIA GPU 时安装 CUDA 版 PyTorch（参考 pytorch.org 的安装命令）；CPU 版 PyTorch 也可以跑通全流程，只是训练较慢。

检查环境：

```bash
python3 check_deps.py
```

安装依赖后、正式训练前可先运行冒烟测试（几秒完成，不需要数据集）：

```bash
python3 test_smoke.py
```

## 准备数据

将 UTF-8 文本放入 `train_dataset/`：

```text
train_dataset/
├── document_001.txt
├── document_002.txt
└── ...
```

建议至少提供两篇文档，否则无法形成独立验证集。当前仓库的 `train_dataset/` 为空，运行训练前必须先加入语料。

## 手动训练

依次执行：

```bash
python3 token_train.py
python3 tokenizer.py
python3 train.py
```

恢复训练：

```bash
python3 train.py --resume
```

查看训练曲线：

```bash
tensorboard --logdir out/runs
```

## Linux 后台一键训练

```bash
chmod +x start.sh
./start.sh
```

脚本通过 `nohup` 启动独立后台进程，关闭 SSH 后训练仍会继续。它会依次清理旧生成物、训练 tokenizer、生成 token 数据并训练模型。

指定虚拟环境 Python：

```bash
PYTHON_BIN=/path/to/venv/bin/python ./start.sh
```

日志写入：

```text
logs/cleanup_<timestamp>.log
logs/token_train_<timestamp>.log
logs/tokenizer_<timestamp>.log
logs/train_<timestamp>.log
logs/pipeline_<timestamp>.log
```

查看流水线状态：

```bash
tail -f logs/pipeline_*.log
```

查看训练 loss：

```bash
tail -f logs/train_*.log
```

## 推理

训练完成后（KV cache 增量解码：预填充一次，之后每步只前向新 token）：

```bash
python3 infer.py "从前有座山"
```

示例参数：

```bash
python3 infer.py "你好" \
  --model-dir ./model_weights \
  --max-gen 200 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.95 \
  --rep-penalty 1.05 \
  --seed 42
```

## 导出与部署（HF / vLLM / SGLang）

模型架构与 `LlamaForCausalLM` 完全同构，可导出为 Hugging Face 标准格式：

```bash
pip install safetensors
python3 export_hf.py --ckpt-dir model_weights --out-dir model_weights_hf
```

导出目录内容：

```text
model_weights_hf/
├── config.json             # LlamaConfig 兼容（architectures: LlamaForCausalLM）
├── generation_config.json  # 采样默认参数
├── model.safetensors       # 权重（tied embeddings 只存一份）
├── tokenizer.model         # SentencePiece 模型
└── tokenizer_config.json   # LlamaTokenizer 声明
```

该目录可直接被 transformers / vLLM / SGLang 加载（后两者需 Linux 环境，不支持原生 Windows）：

```bash
# transformers 加载
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
m = AutoModelForCausalLM.from_pretrained('model_weights_hf'); \
t = AutoTokenizer.from_pretrained('model_weights_hf'); print('OK')"

# vLLM 服务（OpenAI 兼容 API）
vllm serve model_weights_hf

# SGLang 服务
python3 -m sglang.launch_server --model-path model_weights_hf
```

注意：本地自研推理（`infer.py`）使用 `tokenizer.py` 的自定义解码逻辑；HF/vLLM 侧由 SentencePiece 模型内置规则解码，二者对常规文本等价。

## 已知限制

- 上下文窗口 512：RoPE cos/sin 缓存按 `max_len` 预计算；生成时上下文总长达到 max_len 即停止（加长窗口只需调大 `max_len` 重新导出，或配合 NTK/YaRN 缩放外推）。
- 注意力计算复杂度仍为 O(S²)（SDPA 将显存降为 O(S)，但不改变渐近计算量）。
- 本地 KV cache 为朴素 concat 实现，适合单请求；多并发高吞吐请使用 vLLM/SGLang（PagedAttention + continuous batching）。
- 未做 gradient checkpointing 与多卡分布式（第四批，当前百万级参数规模用不到）。
- 新旧 checkpoint 互不兼容（RoPE/GQA 改变了权重布局），需重新训练。
- vLLM/SGLang 不支持原生 Windows，需 Linux / WSL2 / Docker 环境。
- BF16 需要 Ampere 及以上 GPU；旧卡自动回退 FP16，CPU 训练为 FP32。
