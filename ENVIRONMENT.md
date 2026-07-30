# 环境依赖

## 已验证环境

当前实验在以下环境中运行：

| 项目 | 版本 |
| --- | --- |
| 操作系统 | Ubuntu / Linux kernel 5.15，glibc 2.35 |
| Python | 3.10.20 |
| CUDA（PyTorch build） | 12.4 |
| PyTorch | 2.6.0 |
| vLLM | 0.8.5 |
| Transformers | 4.55.4 |
| NumPy | 2.2.6 |
| tqdm | 4.67.3 |
| safetensors | 0.7.0 |
| datasets | 3.6.0 |
| ms-swift 环境 | 4.4.0.dev0 |

`ms-swift` 是原实验所在 Conda 环境的一部分，但当前评测代码不直接依赖 `ms-swift` API，因此没有写入 `requirements.txt`。

## 安装

建议使用独立环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA、PyTorch 和 vLLM 的兼容性与 GPU 型号有关。如果上述精确版本无法安装，应优先参考 vLLM 官方安装说明创建兼容环境，不要在共享服务器上直接覆盖已有环境。

## 硬件

实验需要能够容纳 8B 目标模型或 Guard 模型的 NVIDIA GPU。代码支持：

```bash
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
```

显存不足时可降低三个 batch size：

```bash
GENERATION_BATCH_SIZE=32
SCORING_BATCH_SIZE=32
GUARD_BATCH_SIZE=32
```

正式报告中还应保存：

```bash
python -m pip freeze > environment.freeze.txt
nvidia-smi > nvidia-smi.txt
```

## 数据构造的额外模型

重新构造成对边界数据需要：

- Qwen3-Embedding-4B
- Qwen3-Reranker-4B

运行方法见 [DATA_CARD.md](DATA_CARD.md)。这些模型权重不包含在本仓库中。
