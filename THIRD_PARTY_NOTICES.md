# 第三方来源声明

本项目的 MIT 许可证只覆盖本仓库自行编写的代码，不改变以下第三方项目、数据或模型的许可证。

## 安全推理方法

本仓库不复制三个方法仓库的代码或 prompt。适配器要求用户自行下载官方仓库，并在运行时读取官方入口。

### SafeLLM with Intention Analysis

- 仓库：https://github.com/alphadl/SafeLLM_with_IntentionAnalysis
- 本实验记录的 commit：`58fd692fea1b0ecfddb6db4223e96ab0b456f6d4`
- 使用入口：`demo/IA_demo.py`
- 上游仓库在记录该 commit 时未包含根目录许可证文件。使用者应自行确认其适用条款。

### Goal Prioritization

- 仓库：https://github.com/thu-coai/JailbreakDefense_GoalPriority
- 本实验记录的 commit：`e92c22e125406266ee55058967954895f8091fdf`
- 使用入口：`utils/utils.py::add_defense`
- 上游仓库在记录该 commit 时未包含根目录许可证文件。使用者应自行确认其适用条款。

### SAGE

- 仓库：https://github.com/NJUNLP/SAGE
- 本实验记录的 commit：`24f936ba7b7c05ed0692f60cd320be5c0ae56e26`
- 使用入口：`defense_prompts.py::make_sage_prompt`
- 上游许可证：MIT

## 数据

WildJailbreak：

- 数据卡：https://huggingface.co/datasets/allenai/wildjailbreak
- 许可证：ODC-By 1.0
- Responsible Use Guidelines：https://allenai.org/responsible-use

本仓库中的边界对和拒答诱导集是 WildJailbreak 的派生数据库，继续按照 ODC-By 1.0 提供。

## 模型

本仓库不分发任何模型权重。使用者需要自行获取并遵守相应模型条款：

- Meta-Llama-3.1-8B-Instruct
- Qwen3-8B
- Qwen3Guard-Gen-8B
- Llama-Guard-3-8B
- Qwen3-Embedding-4B
- Qwen3-Reranker-4B
