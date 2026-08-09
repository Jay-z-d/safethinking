# Representation and Separability Analysis

This directory implements the first mechanistic SafeThinking experiment: compare
held-out benign/harmful linear separability before and after explicit safety
reasoning.

## One-time server environment

Use a project-specific environment instead of modifying an alumnus's existing
environments:

```bash
cd /home/lzhao/Work/safethinking
conda create -n safethinking python=3.10 -y
conda activate safethinking
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The shared Llama model is available at
`/home/share/models/Meta-Llama-3.1-8B-Instruct`, so the first experiment does not
need to download model weights.

## Checkpoints

- `h_query`: original user query at the generation boundary.
- `h_guided`: safety-method prompt before generated reasoning.
- `h_analysis_boundary_true`: true IA analysis followed by the fixed end-of-turn
  boundary, before the continuation request.
- `h_preanswer_true`: true IA analysis plus the fixed continuation request and
  assistant generation header, immediately before the visible final answer.
- Matching `*_shuffled` and `*_empty` checkpoints control for non-query-specific
  analysis text and the stage-2 chat structure.

For the first controlled run, use `safe_llm_intention_analysis`: its saved stage-2
messages define the analysis boundary and pre-answer state exactly. Prompt-wrapper methods may lack a separate
reasoning checkpoint when their output does not explicitly delimit analysis from
the final answer.

## Formal pilot

The corrected 100-pair, three-seed workflow is documented in
[`FORMAL_RUNBOOK.md`](FORMAL_RUNBOOK.md). It freezes harmful-source grouped folds,
builds fold-local length-matched controls, extracts multiple layers, and runs
source-clustered refit-bootstrap probes. The older commands below describe the
historical 20-pair smoke workflow only.

After the pilot gates pass, use [`FORMAL_FULL_RUNBOOK.md`](FORMAL_FULL_RUNBOOK.md)
for the immutable 3,343-pair run. Its Slurm arrays are resumable and its merge
steps reject missing, duplicate, checksum-mismatched, or cross-model artifacts.

## 1. Generate a small IA dataset

Clone the official IA method source once (it is intentionally gitignored):

```bash
mkdir -p methods
git clone https://github.com/alphadl/SafeLLM_with_IntentionAnalysis.git \
  methods/SafeLLM_with_IntentionAnalysis
```

Submit 40 normalized input rows (20 benign/harmful pairs), one seed:

```bash
MODEL=/home/share/models/Meta-Llama-3.1-8B-Instruct \
OUTPUT=/home/lzhao/Work/safethinking/outputs/repr-smoke/ia.boundary_generation.jsonl \
sbatch --export=ALL,MODEL,OUTPUT representation_analysis/slurm/generate_ia_smoke.slurm
```

Wait for `COMPLETED` before extracting representations. Generation output is
resumable, but do not reuse the same path with a different model, method, or
experiment definition.

## 2. Extract hidden states

The input is a `boundary_generation.jsonl` produced by the existing pipeline. The
extractor uses a base transformer model rather than a causal-LM head, so it does
not allocate unused vocabulary logits.

```bash
python -m representation_analysis.extract_hidden_states \
  --input outputs/llama31_8b_main_safe_llm_intention_analysis.boundary_generation.jsonl \
  --output-dir outputs/representations/llama31_8b_ia_last \
  --model /home/share/models/Meta-Llama-3.1-8B-Instruct \
  --layers=-1 \
  --pooling last_token \
  --batch-size 2 \
  --dtype bfloat16 \
  --limit 40
```

Outputs are sharded safetensors plus `metadata.jsonl` and `manifest.json`. The
extractor refuses to truncate checkpoints; set `--max-length` to turn an excessive
context into an explicit error.

## 3. Run the source-grouped probe

```bash
python -m representation_analysis.probe_separability \
  --input-dir outputs/representations/llama31_8b_ia_last \
  --output outputs/representations/llama31_8b_ia_last/probe.json \
  --method safe_llm_intention_analysis \
  --before h_query \
  --after h_analysis_boundary_true \
  --folds 5 \
  --bootstrap-samples 2000
```

Both sides, all seeds, and every pair sharing one `harmful_source_index` remain in
the same fold. The scaler is fitted only on before-state training representations
and reused for the after state. Separate probes report fold-aggregated ROC-AUC,
balanced accuracy, and the sign-consistent raw-hidden-space margin
`y * decision / ||w_raw||`. Confidence intervals use a harmful-source cluster
bootstrap that refits preprocessing and probes.

## 4. Slurm smoke run

Create the output parent directory before submitting, then provide paths through
environment variables:

```bash
MODEL=/home/share/models/<model-dir> \
INPUT=/home/lzhao/Work/safethinking/outputs/<ia-generation>.jsonl \
OUTPUT_DIR=/home/lzhao/Work/safethinking/outputs/representations/ia-smoke \
sbatch --export=ALL,MODEL,INPUT,OUTPUT_DIR representation_analysis/slurm/extract_smoke.slurm
```

The smoke script requests one A100 40GB for at most 30 minutes. It does not pin a
node. Check availability and points immediately before submission with
`scir-watch -s` and `scir-account -d`.

Run the probe as a small CPU Slurm job rather than loading the login node:

```bash
INPUT_DIR=/home/lzhao/Work/safethinking/outputs/representations/ia-smoke \
OUTPUT=$INPUT_DIR/probe.json \
METHOD=safe_llm_intention_analysis \
sbatch --export=ALL,INPUT_DIR,OUTPUT,METHOD representation_analysis/slurm/probe.slurm
```
