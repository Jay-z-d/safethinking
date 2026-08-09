# Formal IA representation pilot runbook

This runbook launches the post-fix 100-pair, three-seed pilot. It is not the
3,343-pair main experiment. Run each next stage only after the previous stage
has completed and its validation check passes.

## 0. Common environment

```bash
cd ~/Work/jay-z-d/safethinking-jayzd
conda activate safethinking_jayzd

export PROJECT_ROOT=$PWD
export CONDA_ENV=safethinking_jayzd
export MODEL=/home/share/models/Meta-Llama-3.1-8B-Instruct
export RUN_TAG=llama31-ia-formal-pilot-v1
export RUN_ROOT=$PWD/outputs/$RUN_TAG
mkdir -p "$RUN_ROOT"
```

Before GPU work:

```bash
python -m unittest discover -s tests -v
scir-account -d
scir-watch -s
```

All tests must pass. On the server, the extraction math tests must run rather
than be skipped because the project environment includes PyTorch.

## 1. Freeze data and grouped folds (CPU)

```bash
export INPUT=$PWD/data/boundary/wildjailbreak_pairs.jsonl
export OUTPUT_DIR=$RUN_ROOT/formal-data

sbatch \
  --export=ALL,PROJECT_ROOT,CONDA_ENV,INPUT,OUTPUT_DIR \
  representation_analysis/slurm/prepare_formal_data.slurm
```

After completion:

```bash
cat "$RUN_ROOT/formal-data/manifest.json"
wc -l "$RUN_ROOT/formal-data/pilot_pairs.jsonl"
wc -l "$RUN_ROOT/formal-data/pilot_folds.jsonl"
wc -l "$RUN_ROOT/formal-data/formal_folds.jsonl"
test -f "$RUN_ROOT/formal-data/_SUCCESS" && echo "formal data ready"
```

Expected: 100 pilot pairs, 100 pilot-fold rows, 3,343 formal-fold rows, 1,243
canonical pairs, pilot fold counts 20/20/20/20/20, formal fold pair counts
669/669/669/668/668, and a completion marker.

## 2. No-GPU prompt validation

```bash
python generate_wildjailbreak_responses_vllm.py \
  --input "$RUN_ROOT/formal-data/pilot_pairs.jsonl" \
  --output /tmp/ia-formal-pilot-dry-run.jsonl \
  --model "$MODEL" \
  --method safe_llm_intention_analysis \
  --methods-root methods \
  --seeds 42 43 44 \
  --limit 4 \
  --enable-thinking false \
  --store-prompts \
  --dry-run
```

Expected: four normalized queries and both IA stage previews. A dry run must not
create generation output.

## 3. Generate true IA traces (one GPU)

```bash
export PAIRS=$RUN_ROOT/formal-data/pilot_pairs.jsonl
export OUTPUT=$RUN_ROOT/ia_true.generation.jsonl
export SEEDS="42 43 44"

sbatch \
  --gres=gpu:nvidia_a800_80gb_pcie:1 \
  --export=ALL,PROJECT_ROOT,CONDA_ENV,MODEL,PAIRS,OUTPUT,SEEDS \
  representation_analysis/slurm/generate_ia_pilot.slurm
```

The output is resumable under the same immutable run definition. After the job
leaves `squeue`:

```bash
wc -l "$RUN_ROOT/ia_true.generation.jsonl"
tail -n 50 slurm-st-ia-pilot-<JOBID>.out
tail -n 50 slurm-st-ia-pilot-<JOBID>.err
```

Expected: 100 pairs x 2 sides x 3 seeds = 600 rows, with no traceback.

## 4. Build fold-local shuffled/empty controls (CPU)

```bash
export INPUT=$RUN_ROOT/ia_true.generation.jsonl
export OUTPUT=$RUN_ROOT/ia_controlled.generation.jsonl
export FOLD_MANIFEST=$RUN_ROOT/formal-data/pilot_folds.jsonl
export TOKENIZER=$MODEL

sbatch \
  --export=ALL,PROJECT_ROOT,CONDA_ENV,INPUT,OUTPUT,FOLD_MANIFEST,TOKENIZER \
  representation_analysis/slurm/build_controls_pilot.slurm
```

After completion:

```bash
wc -l "$RUN_ROOT/ia_controlled.generation.jsonl"
tail -n 50 slurm-st-ia-controls-<JOBID>.out
tail -n 50 slurm-st-ia-controls-<JOBID>.err
```

Expected: 600 rows. Every row must have `representation_controls`; shuffled
donors stay in the same fold and seed, use a different source group, have donor
labels balanced independently of recipient labels. Donor text is composed when
needed and truncated so every shuffled analysis has exactly the same tokenizer
token count as its corresponding true analysis.

## 5. Extract hidden states (one GPU)

```bash
export INPUT=$RUN_ROOT/ia_controlled.generation.jsonl
export OUTPUT_DIR=$RUN_ROOT/representations
export LAYERS="0,4,8,12,16,20,24,28,32"

sbatch \
  --gres=gpu:nvidia_a800_80gb_pcie:1 \
  --export=ALL,PROJECT_ROOT,CONDA_ENV,MODEL,INPUT,OUTPUT_DIR,LAYERS \
  representation_analysis/slurm/extract_formal_pilot.slurm
```

After completion:

```bash
cat "$RUN_ROOT/representations/manifest.json"
test -f "$RUN_ROOT/representations/_SUCCESS" && echo "representations ready"
```

Expected: 600 rows x eight checkpoints = 4,800 checkpoint representations.
The eight checkpoints are query, guided, and true/shuffled/empty analysis and
pre-answer states. Nine layer columns are stored for each checkpoint.

## 6. Run the two primary probes (CPU)

The first contrast tests reasoning versus the guided prompt. Start with seed 42:

```bash
export INPUT_DIR=$RUN_ROOT/representations
export FOLD_MANIFEST=$RUN_ROOT/formal-data/pilot_folds.jsonl
export METHOD=safe_llm_intention_analysis
export BEFORE=h_guided
export AFTER=h_analysis_boundary_true
export RUN_ID=42
export OUTPUT=$RUN_ROOT/probe.guided-to-true.seed42.json
export BOOTSTRAP_SAMPLES=200

sbatch \
  --export=ALL,PROJECT_ROOT,CONDA_ENV,INPUT_DIR,FOLD_MANIFEST,METHOD,BEFORE,AFTER,RUN_ID,OUTPUT,BOOTSTRAP_SAMPLES \
  representation_analysis/slurm/probe_formal.slurm
```

The second contrast tests true reasoning content versus a matched shuffled trace:

```bash
export BEFORE=h_analysis_boundary_shuffled
export AFTER=h_analysis_boundary_true
export RUN_ID=42
export OUTPUT=$RUN_ROOT/probe.shuffled-to-true.seed42.json

sbatch \
  --export=ALL,PROJECT_ROOT,CONDA_ENV,INPUT_DIR,FOLD_MANIFEST,METHOD,BEFORE,AFTER,RUN_ID,OUTPUT,BOOTSTRAP_SAMPLES \
  representation_analysis/slurm/probe_formal.slurm
```

If both jobs complete, repeat them with `RUN_ID=43` and `RUN_ID=44`, using new
output filenames. Then run `RUN_ID=all` as the grouped hierarchical summary.
Never reuse an output path.

Probe results must report:

- `group_field: source_group`;
- `scaler_fit: before_train_only`;
- `margin_space: raw_hidden_state`;
- fold-wise metrics;
- `bootstrap_source_refit_95_ci`;
- `bootstrap_duplicate_source_policy: same_original_source_same_fold`.

## 7. Budget gate

Record job IDs, elapsed time, output token counts, and account points before and
after generation/extraction. Project the 3,343-pair, three-seed cost from this
pilot. Keep at least 150 points unused and do not launch the main experiment if
the projected additional cost exceeds 700 points.

The main run requires a new `RUN_TAG`, immutable manifest, and wall-time/sharding
settings derived from this pilot. Do not obtain the main result by merely raising
the old smoke script's `LIMIT`.
