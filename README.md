# SafeThinking Boundary Evaluation

本仓库提供一个面向成对安全边界数据的安全推理评测框架。框架不要求被测模型自行输出安全标签，而是收集 Base 模型和安全推理方法的实际回复、显式方法 trace，并分别计算：

- `refusal_score`：模型在生成上下文中的拒答倾向。
- `harmful_score`：最终回复的有害程度。

最终只报告三个连续指标：

| 指标 | 方向 | 含义 |
| --- | ---: | --- |
| Intent Error | ↓ | 模型是否对 benign 输入形成回答意图、对 harmful 输入形成拒答意图 |
| Harmful Outcome | ↓ | harmful 输入最终产生的有害内容程度 |
| Behavioral Boundary Margin | ↑ | 成对 benign/harmful 输入之间的联合行为区分度 |

## 仓库内容

```text
data/
  boundary/wildjailbreak_pairs.jsonl
  refusal_induction/vanilla_harmful_500.jsonl
  refusal_templates/
data_construction/
  build_wildjailbreak_pairs.py
response_generation/
  methods/
  run_core_pipeline.sh
  ...
OPEN_SOURCE_RUN_GUIDE.md
ENVIRONMENT.md
DATA_CARD.md
THIRD_PARTY_NOTICES.md
requirements.txt
```

当前支持：

- Direct/Base
- SafeLLM with Intention Analysis
- Goal Prioritization
- SAGE

## 快速开始

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

下载三个方法的官方仓库：

```bash
mkdir -p methods
git clone https://github.com/alphadl/SafeLLM_with_IntentionAnalysis.git methods/SafeLLM_with_IntentionAnalysis
git clone https://github.com/thu-coai/JailbreakDefense_GoalPriority.git methods/JailbreakDefense_GoalPriority
git clone https://github.com/NJUNLP/SAGE.git methods/SAGE
```

运行 Llama-3.1 示例：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL=/path/to/Meta-Llama-3.1-8B-Instruct \
MODEL_TAG=llama31_8b \
PAIRS=data/boundary/wildjailbreak_pairs.jsonl \
REFUSAL_INDUCTION=data/refusal_induction/vanilla_harmful_500.jsonl \
QWEN_GUARD_MODEL=/path/to/Qwen3Guard-Gen-8B \
LLAMA_GUARD_MODEL=/path/to/Llama-Guard-3-8B \
ENABLE_THINKING=auto \
bash response_generation/run_core_pipeline.sh
```

完整的数据格式、方法 commit、Qwen3 运行方式、拒答模板诱导流程、校准方式和指标公式见 [中文端到端复现指南](OPEN_SOURCE_RUN_GUIDE.md)。

## 数据来源与安全警告

本仓库数据派生自 [AllenAI WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak)，包含可能令人不适的有害、欺骗、仇恨、自伤和网络攻击类文本，仅用于安全研究与评测。

WildJailbreak 数据库以 [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) 提供，并受 [AI2 Responsible Use Guidelines](https://allenai.org/responsible-use) 约束。数据的详细来源、构造过程、引用和校验和见 [DATA_CARD.md](DATA_CARD.md) 和 [data/LICENSE-ODC-BY.txt](data/LICENSE-ODC-BY.txt)。

## 许可证

- 本项目自行编写的代码：MIT，见 [LICENSE](LICENSE)。
- `data/` 中派生数据库：ODC-By 1.0。
- 第三方方法、模型和原始数据不受本项目 MIT 许可证覆盖，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
