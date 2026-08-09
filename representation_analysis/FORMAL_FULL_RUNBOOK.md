# Frozen 3,343-Pair Formal Runbook

This runbook starts only after the 100-pair pilot gates pass. Scientific choices
are frozen at commit `31b05ad`: Llama-3.1-8B-Instruct, IA prompts, seeds
42/43/44, five harmful-source-grouped folds, final-layer logistic probes,
before-train-only scaling, raw-space normalized margins, and source-cluster
refit bootstrap. Later commits may add only deterministic sharding, validation,
and recovery machinery.

Run each numbered stage only after the preceding stage has produced its declared
`_SUCCESS` marker or validated output manifest. Never merge partial array jobs.

## 0. Immutable paths

```bash
cd ~/Work/jay-z-d/safethinking-jayzd
conda activate safethinking_jayzd

export PROJECT_ROOT=$PWD
export CONDA_ENV=safethinking_jayzd
export MODEL=/home/share/models/Meta-Llama-3.1-8B-Instruct
export RUN_TAG=llama31-ia-formal-full-v1
export FULL_ROOT=$PWD/outputs/$RUN_TAG
mkdir -p "$FULL_ROOT"
git rev-parse HEAD | tee "$FULL_ROOT/code_commit.txt"
```

Record `scir-account -d` before and after both GPU array stages. Keep at least
150 points unused. GPU arrays default to one simultaneous A800 (`%1`).

## 1. Freeze formal folds again under the full-run tag

```bash
export INPUT=$PWD/data/boundary/wildjailbreak_pairs.jsonl
export OUTPUT_DIR=$FULL_ROOT/formal-data

sbatch --export=ALL,PROJECT_ROOT,CONDA_ENV,INPUT,OUTPUT_DIR \
  representation_analysis/slurm/prepare_formal_data.slurm
```

Gate: `formal-data/_SUCCESS`; manifest counts are 3,343 pairs, 1,243 harmful
sources, and fold pair counts 669/669/669/668/668.

## 2. Make 14 immutable pair shards

```bash
export FOLD_MANIFEST=$FULL_ROOT/formal-data/formal_folds.jsonl
export OUTPUT_DIR=$FULL_ROOT/pair-shards
export PAIRS_PER_SHARD=250

sbatch --export=ALL,PROJECT_ROOT,CONDA_ENV,INPUT,FOLD_MANIFEST,OUTPUT_DIR,PAIRS_PER_SHARD \
  representation_analysis/slurm/prepare_full_shards.slurm
```

Gate: `pair-shards/_SUCCESS`; manifest reports 14 shards and 3,343 pairs.

## 3. Generate IA traces as a resumable GPU array

```bash
export PAIR_SHARDS_DIR=$FULL_ROOT/pair-shards
export GENERATION_DIR=$FULL_ROOT/generation-shards
export SEEDS="42 43 44"

sbatch --array=0-13%1 \
  --gres=gpu:nvidia_a800_80gb_pcie:1 \
  --export=ALL,PROJECT_ROOT,CONDA_ENV,MODEL,PAIR_SHARDS_DIR,GENERATION_DIR,SEEDS \
  representation_analysis/slurm/generate_ia_full_shard.slurm
```

Each completed 250-pair shard has 1,500 rows; the final 93-pair shard has 558.
Generation files are append-resumable. Retry only failed indices with
`--array=<index>`; do not delete completed shard files.

## 4. Strictly merge and validate all 20,058 generations

```bash
export OUTPUT=$FULL_ROOT/ia_true.generation.jsonl

sbatch --export=ALL,PROJECT_ROOT,CONDA_ENV,MODEL,PAIR_SHARDS_DIR,GENERATION_DIR,OUTPUT \
  representation_analysis/slurm/merge_full_generations.slurm
```

Gate: `ia_true.generation.jsonl._SUCCESS`; the merge rejects missing or duplicate
pair/side/seed keys, model mismatches, method mismatches, source-group mismatches,
and altered pair-shard checksums.

## 5. Build exact-length controls over the complete dataset

```bash
export INPUT=$FULL_ROOT/ia_true.generation.jsonl
export OUTPUT=$FULL_ROOT/ia_controlled.generation.jsonl
export TOKENIZER=$MODEL

sbatch --export=ALL,PROJECT_ROOT,CONDA_ENV,INPUT,OUTPUT,FOLD_MANIFEST,TOKENIZER \
  representation_analysis/slurm/build_controls_full.slurm
```

Expected: 20,058 rows, 3,343 pairs, three runs, five folds, and true/shuffled/
empty controls. This stage intentionally uses the complete fold pools before
any representation sharding.

## 6. Split controlled rows back onto the frozen pair shards

```bash
export INPUT=$FULL_ROOT/ia_controlled.generation.jsonl
export OUTPUT_DIR=$FULL_ROOT/controlled-shards

sbatch --export=ALL,PROJECT_ROOT,CONDA_ENV,INPUT,PAIR_SHARDS_DIR,OUTPUT_DIR \
  representation_analysis/slurm/split_full_controls.slurm
```

Gate: `controlled-shards/_SUCCESS`; total rows are 20,058.

## 7. Extract 160,464 representations as a GPU array

```bash
export CONTROLLED_SHARDS_DIR=$FULL_ROOT/controlled-shards
export REPRESENTATION_ROOT=$FULL_ROOT/representation-shards
export LAYERS="0,4,8,12,16,20,24,28,32"

sbatch --array=0-13%1 \
  --gres=gpu:nvidia_a800_80gb_pcie:1 \
  --export=ALL,PROJECT_ROOT,CONDA_ENV,MODEL,CONTROLLED_SHARDS_DIR,REPRESENTATION_ROOT,LAYERS \
  representation_analysis/slurm/extract_full_shard.slurm
```

Completed shard directories are skipped safely. A partial directory is never
overwritten; move that exact `repr-<index>` directory aside before retrying the
failed index.

## 8. Hard-link and validate representation shards

```bash
export OUTPUT_DIR=$FULL_ROOT/representations

sbatch --export=ALL,PROJECT_ROOT,CONDA_ENV,CONTROLLED_SHARDS_DIR,REPRESENTATION_ROOT,OUTPUT_DIR \
  representation_analysis/slurm/merge_full_representations.slurm
```

Gate: `representations/_SUCCESS`; manifest reports 160,464 checkpoints and each
of the eight checkpoint types appears 20,058 times. Hard links avoid duplicating
the approximately 12 GB tensor payload when the filesystem supports them.

## 9. Run the two 1,000-draw primary bootstraps in deterministic chunks

Common variables:

```bash
export INPUT_DIR=$FULL_ROOT/representations
export METHOD=safe_llm_intention_analysis
export OUTPUT_DIR=$FULL_ROOT/probe-chunks
export BOOTSTRAP_CHUNK_SIZE=50
export LAYER_COLUMN=-1
```

Reasoning-gain contrast:

```bash
export BEFORE=h_guided
export AFTER=h_analysis_boundary_true
export OUTPUT_PREFIX=guided-to-true

sbatch --array=0-19%2 \
  --export=ALL,PROJECT_ROOT,CONDA_ENV,INPUT_DIR,FOLD_MANIFEST,METHOD,BEFORE,AFTER,OUTPUT_DIR,OUTPUT_PREFIX,BOOTSTRAP_CHUNK_SIZE,LAYER_COLUMN \
  representation_analysis/slurm/probe_full_bootstrap_shard.slurm
```

Content-specific contrast:

```bash
export BEFORE=h_analysis_boundary_shuffled
export AFTER=h_analysis_boundary_true
export OUTPUT_PREFIX=shuffled-to-true

sbatch --array=0-19%2 \
  --export=ALL,PROJECT_ROOT,CONDA_ENV,INPUT_DIR,FOLD_MANIFEST,METHOD,BEFORE,AFTER,OUTPUT_DIR,OUTPUT_PREFIX,BOOTSTRAP_CHUNK_SIZE,LAYER_COLUMN \
  representation_analysis/slurm/probe_full_bootstrap_shard.slurm
```

Chunk zero for each contrast also saves the shared fixed-fold OOF predictions.
Every other chunk suppresses the redundant prediction file.

## 10. Merge exact bootstrap draw sequences

```bash
export INPUT_DIR=$FULL_ROOT/probe-chunks
export CHUNKS=20

export INPUT_PREFIX=guided-to-true
export OUTPUT=$FULL_ROOT/probe.guided-to-true.all-seeds.final.json
sbatch --export=ALL,PROJECT_ROOT,CONDA_ENV,INPUT_DIR,INPUT_PREFIX,OUTPUT,CHUNKS \
  representation_analysis/slurm/merge_full_probes.slurm

export INPUT_PREFIX=shuffled-to-true
export OUTPUT=$FULL_ROOT/probe.shuffled-to-true.all-seeds.final.json
sbatch --export=ALL,PROJECT_ROOT,CONDA_ENV,INPUT_DIR,INPUT_PREFIX,OUTPUT,CHUNKS \
  representation_analysis/slurm/merge_full_probes.slurm
```

Final gates for both JSON files: 20,058 samples, 3,343 pairs, 1,243 source
groups, 1,000 bootstrap draws, `bootstrap_sharded_equivalent: true`, and
`bootstrap_duplicate_source_policy: same_original_source_same_fold`.
