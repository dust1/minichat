# MiniChat：手写 Decoder-only Transformer 学习项目

MiniChat 是一个使用 NumPy/CuPy 手写前向传播、反向传播、优化器和训练循环的自回归语言模型学习项目。它的目标是展示一个小型 GPT 类模型从 tokenizer 训练、数据切分、next-token 训练、验证到文本生成的完整链路。

> 受资源限制，在20MB的中文预料上完成过训练，能够正常推理出东西，但是属于前言不搭后语的程度，不过至少能说话不会乱码

## 分支记录
learn_1 - 最初版本，使用Numpy/cupy手动实现的小写GPT类模型


## 当前完成状态

第一层规划中的主要代码能力已经实现：

| 修改点 | 实现状态 | 说明 |
|---|---|---|
| Transformer Block 正确反向传播 | 已实现 | 按 FFN → Attention 的逆序传播，并处理两级残差分支 |
| 训练与推理统一 tokenizer | 已实现 | 均使用 SentencePiece BBPE；推理从 checkpoint 加载 tokenizer |
| 空格、下划线和 Unicode 可逆 | 已实现 | 直接编码 Unicode，并支持 byte fallback |
| 小数据训练步数保护 | 已实现 | batch 数使用向上取整，warmup 和余弦退火有边界保护 |
| EOS、文档边界、padding 与 mask | 已实现 | 窗口不跨文档，尾部 padding 的 target 使用 `ignore_index=-1` |
| 验证集、验证 loss 与 perplexity | 已实现 | 验证 loss 按有效 token 数加权 |
| 模型配置、tokenizer 与 checkpoint | 已实现 | 保存模型参数、AdamW 状态、模型配置和 tokenizer |
| AdamW 与全局梯度裁剪 | 已实现 | 手写 AdamW，一次性计算全模型梯度范数 |
| 可控采样 | 已实现 | 支持 temperature、top-k、top-p、重复惩罚和 seed |

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
Token Embedding + Learned Position Embedding
      │
      ▼
4 × Pre-LN Transformer Block
      │
      ├─ LayerNorm → Causal Self-Attention → Dropout → Residual
      │
      └─ LayerNorm → Linear → ReLU → Linear → Dropout → Residual
      │
      ▼
Final LayerNorm
      │
      ▼
Vocabulary Classifier
      │
      ▼
Next-token Cross Entropy
```

### 默认模型配置

默认超参数定义在 `train.py`：

| 参数 | 默认值 |
|---|---:|
| 架构 | Decoder-only Transformer |
| Transformer Block | 4 |
| 隐藏维度 `d_model` | 128 |
| FFN 维度 `d_ff` | 256 |
| 注意力 | 单头 causal self-attention |
| 最大上下文 | 512 tokens |
| Dropout | 0.1 |
| Batch size | 64 |
| Epochs | 30 |
| 基础学习率 | 3e-4 |
| Warmup | 最多 200 steps，且不超过总步数一半 |
| 最低学习率 | 1e-5 |
| 优化器 | 手写 AdamW |
| Weight decay | 0.1 |
| 全局梯度裁剪 | 1.0 |
| 数据窗口 stride | 256 |

词表大小由语料规模动态决定，范围为 2,000～32,000。默认架构的参数量约为：

```text
参数量 = 529,664 + 257 × vocab_size
```

例如词表为 2,000 时，模型约有 104 万参数。

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

### 3. Transformer

模型采用 Pre-LN 残差结构。每个 Block 包含：

1. LayerNorm；
2. 单头因果自注意力；
3. attention dropout 和残差连接；
4. LayerNorm；
5. 两层 ReLU FFN；
6. FFN dropout 和残差连接。

因果 mask 禁止当前位置关注未来 token。反向传播严格按正向计算图逆序执行：先反传 FFN 子层，再反传 Attention 子层。

### 4. 损失函数

训练目标为标准 next-token cross entropy。项目提供分块融合的 classifier + cross entropy 实现，避免一次性物化完整 `(batch, sequence, vocabulary)` logits，从而降低大词表下的峰值内存。

Padding target 不参与 loss 或梯度，loss 按有效 token 数归一化。

### 5. 优化器和学习率

项目手写 AdamW，维护每个参数的一阶矩和二阶矩，并使用 bias correction。所有模型梯度统一计算全局 L2 norm，超过阈值后按同一比例缩放。

学习率策略：

```text
线性 warmup → cosine decay → min_lr
```

小数据集下使用 `ceil(samples / batch_size)` 计算每轮 batch 数，避免总步数变成 0 或遗漏最后一个不完整 batch。

### 6. Checkpoint

`model_weights/` 中保存：

- 所有模型权重；
- AdamW 一阶矩和二阶矩；
- `config.json` 模型结构和训练进度；
- scheduler step；
- NumPy RNG 状态；
- `bbpe.model` 和 `bbpe.vocab`。

`python train.py --resume` 可从 checkpoint 恢复权重、优化器步数和学习率进度。CPU/NumPy 路径可以恢复随机状态；CuPy RNG 不保证逐随机位精确恢复。

## 项目结构

```text
minichat/
├── train_dataset/       # 原始 UTF-8 .txt 训练语料
├── out/
│   ├── bbpe_tokenizer/  # SentencePiece 模型
│   └── tokenized_data/  # train/val 二进制 token 数据
├── model_weights/       # checkpoint、配置和 tokenizer
├── logs/                # start.sh 运行后生成的阶段日志
├── token_train.py       # 训练 BBPE tokenizer
├── tokenizer.py         # 数据 token 化与 train/val 划分
├── dataloader.py        # 文档窗口和 batch 构造
├── train.py             # 模型、反向传播、优化器和训练循环
├── infer.py             # 文本生成入口
├── check_deps.py        # 环境依赖检查
└── start.sh             # Linux 后台一键训练脚本
```

## 环境准备

最低依赖：

```bash
python3 -m pip install numpy sentencepiece
```

使用 NVIDIA GPU 时，根据本机 CUDA 版本安装对应的 CuPy，例如：

```bash
python3 -m pip install cupy-cuda12x
```

检查环境：

```bash
python3 check_deps.py
```

未安装 CuPy 时会自动退回 NumPy CPU，但训练速度会明显降低。

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

训练完成后：

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

## 已知限制

- Attention 为单头实现，并且没有独立的 attention output projection。
- 使用可学习绝对位置 embedding，上下文固定为 512。
- 注意力复杂度为 `O(sequence²)`。
- 推理没有 KV cache，每生成一个 token 都重新计算整个上下文。
- AdamW 当前也会对 bias 和 LayerNorm 参数施加 weight decay。
- 最佳验证权重和最终权重写入同一个目录，最终保存可能覆盖此前的最佳 checkpoint。
- CuPy 随机数状态不保证精确恢复。
- 当前缺少持续保留的梯度检查和小批次过拟合测试脚本。