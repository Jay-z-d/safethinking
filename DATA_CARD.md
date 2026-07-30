# 数据卡

## 概览

本仓库包含三类数据：

| 文件 | 数量 | 用途 |
| --- | ---: | --- |
| `data/boundary/wildjailbreak_pairs.jsonl` | 3343 对 | 成对安全边界评测 |
| `data/refusal_induction/vanilla_harmful_500.jsonl` | 500 条 | 诱导 Base 模型拒答模板 |
| `data/refusal_templates/*.json` | 每个模型 50 个模板 | 计算共享 refusal logprob |

数据包含有害和对抗性文本，只适合安全研究、受控评测和防御方法开发。

## 来源与许可证

所有输入文本来自 [AllenAI WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak)：

> Contains information from WildJailbreak, which is made available under the Open Data Commons Attribution License (ODC-By) v1.0.

- 数据库许可证：https://opendatacommons.org/licenses/by/1-0/
- AI2 Responsible Use Guidelines：https://allenai.org/responsible-use
- 论文：https://arxiv.org/abs/2406.18510

使用这些文件时必须保留以上来源和许可证声明。

建议引用：

```bibtex
@misc{wildteaming2024,
  title={WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models},
  author={Jiang, Liwei and Rao, Kavel and Han, Seungju and Ettinger, Allyson and Brahman, Faeze and Kumar, Sachin and Mireshghallah, Niloofar and Lu, Ximing and Sap, Maarten and Choi, Yejin and Dziri, Nouha},
  year={2024},
  eprint={2406.18510},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

## 成对边界数据

### 输入

使用 WildJailbreak training split 中：

- `adversarial_benign`
- `adversarial_harmful`

### 构造过程

构造脚本为：

```text
data_construction/build_wildjailbreak_pairs.py
```

默认流程：

1. 使用 Qwen3-Embedding-4B 编码 adversarial benign 和 adversarial harmful。
2. 对每条 benign 输入执行精确 cosine top-50 harmful 检索。
3. 使用 Qwen3-Reranker-4B 对候选对重新排序。
4. 保留 reranker score 不低于 `0.5` 的候选。
5. 每条 benign 输入保留最高分的一个 harmful 候选。
6. 默认不要求 harmful 侧全局唯一。

运行示例：

```bash
python data_construction/build_wildjailbreak_pairs.py \
  --dataset /path/to/wildjailbreak/train.tsv \
  --embedding-model /path/to/Qwen3-Embedding-4B \
  --reranker-model /path/to/Qwen3-Reranker-4B \
  --work-dir wildjailbreak_pair_work \
  --output data/boundary/wildjailbreak_pairs.jsonl \
  --text-field adversarial \
  --top-k 50 \
  --reranker-threshold 0.5 \
  --max-pairs-per-benign 1
```

每行字段包括：

```text
pair_id
benign_source_index
harmful_source_index
benign_label
harmful_label
benign_prompt
harmful_prompt
benign_vanilla
harmful_vanilla
embedding_score
embedding_rank
reranker_score
reranker_rank
```

## 拒答诱导数据

`vanilla_harmful_500.jsonl` 是从 WildJailbreak training split 中按源文件顺序提取的前 500 条 `vanilla_harmful` 输入。

构建命令：

```bash
python -m response_generation.prepare_refusal_induction \
  --input-tsv /path/to/wildjailbreak/train.tsv \
  --output data/refusal_induction/vanilla_harmful_500.jsonl \
  --data-type vanilla_harmful \
  --limit 500
```

## 已构造的拒答模板

仓库提供：

- `llama31_8b_base_induced.json`
- `qwen3_8b_no_thinking_base_induced.json`

两套模板均使用对应目标模型的 Base/Direct 回复生成：

- 输入：500 条 `vanilla_harmful`
- seed：42
- 仅保留回复开头带拒答 cue 的样本
- 每个片段最少 4 个、最多 24 个空白分隔词
- 按完全相同片段的出现频率排序
- 每个模型保留 50 个模板

Llama 模板从 500 条回复中的 424 条拒答式回复构造；Qwen3 no-thinking 模板从 193 条拒答式回复构造。

同一个目标模型上的 Direct、IA、Goal Prioritization 和 SAGE 必须共享对应的 Base 模板。

## 校验和

SHA-256 校验和保存在：

```text
data/CHECKSUMS.sha256
```

可以运行：

```bash
shasum -a 256 -c data/CHECKSUMS.sha256
```

## 限制

- Pair 是模型检索和重排得到的语义近邻，不保证每一对都不存在标注或语义误差。
- Reranker score 不是安全标签，也不是最终评测分数。
- 数据包含多种语言，但构造和模板片段规则主要面向英文。
- 拒答模板会受到目标模型版本、chat template、seed 和生成参数影响。
- 数据不能用于实施、促进或自动化有害行为。
