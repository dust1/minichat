# MiniChat：Decoder-only Transformer 学习项目

MiniChat 是一个小型自回归语言模型学习项目，覆盖从 tokenizer 训练、数据切分、next-token 训练、验证到文本生成的完整链路。

已经改成现代训练框架，不再手写传播

> 受资源限制，在20MB的中文语料上完成过训练，能够正常推理出东西，但是属于前言不搭后语的程度，不过至少能说话不会乱码

## 分支记录
learn_1 - 最初版本，使用Numpy/cupy手动实现的小写GPT类模型
learn_2 - 使用现代化框架版本

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
