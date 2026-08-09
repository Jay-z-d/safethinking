# SafeThinking formal experiment plan

## 1. Research question

Test whether query-specific safety reasoning makes paired benign and harmful inputs more linearly separable in the target model's hidden representation, and whether any representation change corresponds to a better safety-helpfulness boundary rather than indiscriminate refusal.

The first formal experiment is deliberately narrow:

- target model: Meta-Llama-3.1-8B-Instruct;
- method: SafeLLM with Intention Analysis (IA);
- intervention: true query-specific IA reasoning versus matched controls;
- evaluation unit: an official benign/harmful boundary pair, with statistical dependence clustered by `harmful_source_index`;
- generation seeds: 42, 43, and 44.

Native reasoning models, additional inference-time methods, and training-based methods are follow-up experiments. They must reuse the frozen measurement pipeline rather than changing it after the first result is observed.

## 2. Confirmatory hypotheses

### Primary contrasts

1. **Reasoning gain:** held-out AUC at the true analysis boundary is higher than AUC at the IA-guided prompt boundary.
2. **Content-specific gain:** held-out AUC after the true query-specific analysis is higher than AUC after a non-corresponding, token-length-matched shuffled analysis rendered in the identical chat format.

Both contrasts are needed. A positive first contrast without a positive second contrast is compatible with extra tokens or formatting rather than useful reasoning.

### Key secondary contrasts

- End-to-end separability: pre-answer true-IA state versus original-query state.
- Balanced accuracy and fixed-boundary signed margin for all primary comparisons.
- Benign and harmful class-specific margins.
- Behavioral Boundary Margin, Intent Error, and Harmful Outcome for Direct, true IA, shuffled-analysis IA, and empty-analysis IA.
- Association between per-pair representation improvement and per-pair behavioral improvement.

The directional hypothesis is positive, but report two-sided 95% confidence intervals. A negative or null result is a valid outcome.

## 3. Data audit and evaluation population

The current UTF-8 `data/boundary/wildjailbreak_pairs.jsonl` has:

- 3,343 valid JSON rows;
- 3,343 unique benign sources and 1,243 unique harmful sources;
- 401 harmful sources reused in multiple pairs;
- one harmful source reused 656 times.

This is consistent with the original data contract in `DATA_CARD.md`: each benign input keeps its best harmful candidate, while harmful-side global uniqueness is not required. Reuse is therefore part of the official 3,343-pair evaluation distribution, not a corrupt-data error. However, treating the rows as statistically independent would leak exact harmful prompts across probe folds and overstate precision.

### Primary population and dependence control

1. Preserve all 3,343 official boundary pairs for the primary pair-weighted behavioral and representation results.
2. Build grouped folds by `harmful_source_index`; move the whole pair (both benign and harmful sides) with that group. The same harmful prompt must never appear in both probe training and test folds.
3. Balance grouped folds by pair count. Each group contains complete benign/harmful pairs, so class counts remain matched.
4. Use a harmful-source cluster bootstrap for confidence intervals. Resampling individual pair rows is invalid under harmful-side reuse.
5. Report a source-balanced robustness analysis in addition to the official pair-weighted result.
6. Report a canonical one-pair-per-harmful-source sensitivity analysis. Freeze the selection rule before outcomes: highest reranker score, then highest embedding score, then lexicographically smallest `pair_id`.
7. The 20-pair smoke run is disclosed as development work. Freeze all formal choices before examining any additional full-data outcomes.

The pair-weighted result answers the original project's 3,343-boundary-pair question. The source-balanced and canonical analyses answer whether that result is robust to harmful-source reuse. Neither replaces the other.

### Development pilot and formal dataset

- Development pilot: 100 pairs sampled from 100 distinct harmful sources, three generation seeds.
- Formal run: all 3,343 official pairs, three generation seeds, using the frozen configuration.
- Robustness run: the same saved outputs reweighted by source and evaluated on the frozen canonical one-per-harmful subset; no new generation is required.

Create and checksum a separately balanced 100-pair pilot-fold manifest, the grouped 3,343-pair formal-fold manifest, and the canonical sensitivity manifest. All conditions, checkpoints, generation seeds, and probes must use the manifest for their own dataset; do not project the highly imbalanced full-data source groups onto the pilot subset.

Before the main run, manually audit a blinded sample of at least 100 selected pairs for label correctness, semantic relatedness, and accidental near-duplicates. Report the audit protocol and disagreement rate.

## 4. Intervention and checkpoint design

### Saved conditions

For every query and generation seed, construct:

1. `direct`: no IA prompt or analysis;
2. `ia_true`: the genuine query-specific IA analysis;
3. `ia_shuffled`: a deranged donor analysis from another pair, selected independently of recipient label and matched within a predeclared token-length tolerance;
4. `ia_empty`: the identical stage-2 chat skeleton with no analysis content.

The shuffled donor must come from the same grouped fold, must never be the recipient's own analysis, and must be balanced so donor label cannot predict recipient label. Record the donor ID and token-length difference. Construct donors after freezing folds and never shuffle across train/test folds.

### Exact checkpoints

- `h_query`: original query prompt boundary;
- `h_guided`: IA stage-1 guided prompt boundary, before generation;
- `h_analysis_boundary`: immediately after the analysis and its fixed end-of-turn boundary token, before the continuation request;
- `h_preanswer`: after the fixed continuation request and assistant-generation header, before the visible final answer;
- `h_answer_end` (behavioral diagnostic only): final response boundary.

The primary analysis checkpoint must end on the same fixed boundary token in true and control conditions. Also save token-span offsets for the original query, guidance, generated analysis, continuation prompt, and final answer. This prevents the identity of a variable last token from becoming the signal.

## 5. Representation extraction

- Use exact saved token IDs or losslessly reconstructed messages; verify token equality before extraction.
- Reject truncation instead of silently shortening a sample.
- Record tokenizer, chat-template hash, model path and revision, dtype, and maximum length.
- Extract all transformer layers once if storage permits. Select one primary layer on the 100-pair development pilot and freeze it before running the 3,343-pair analysis.
- Primary pooling: fixed boundary-token state.
- Secondary pooling: mean state over the query span or analysis span with padding masked.
- Deduplicate deterministic `h_query` and `h_guided` representations across generation seeds.
- Analyze each generation seed separately and aggregate effects over seeds; do not count repeated seeds or repeated harmful sources as independent observations.

Layer-wise curves are exploratory. Apply false-discovery-rate correction and do not promote a new primary layer after looking at the 3,343-pair results.

## 6. Probe and statistical protocol

### Two complementary probes

1. **Separate decodability probes:** fit one regularized linear model per checkpoint under identical preprocessing, folds, and frozen hyperparameters. ROC-AUC is the primary separability metric.
2. **Transport probe:** fit a single probe on `h_query` training states and apply it unchanged to later checkpoints. This tests whether reasoning moves examples farther along the original safety direction.

### Preprocessing and validation

- Fit the scaler on `h_query` training states only, then reuse that transform for every checkpoint and condition.
- Never fit preprocessing on validation/test states or jointly on a new after-state distribution.
- Freeze logistic/SVM family, `C`, primary layer, and pooling using only the development partition.
- Enforce complete benign/harmful samples and identical checkpoint keys for every `(pair_id, run_id)`.
- Keep every `harmful_source_index` group, all associated pairs, and all seeds/conditions in the same fold.
- Use the exact same fold assignments for all comparisons.
- Report raw-space normalized margin when claiming geometric distance. If a standardized-space margin is also retained, name it explicitly.

### Inference

- On all 3,343 pairs, use five harmful-source-grouped outer folds with frozen hyperparameters to produce paired out-of-fold predictions.
- Build 95% confidence intervals with a harmful-source cluster outer bootstrap that refits preprocessing and probes. Use at least 1,000 resamples for the final table.
- Aggregate repeated generation seeds and pairs within the resampled harmful-source hierarchy.
- Apply Holm correction to the two primary contrasts. Use Benjamini-Hochberg FDR for layer-wise and other exploratory analyses.
- Include label-permutation, random-direction, prompt-length, and bag-of-words baselines.
- Report effect sizes and confidence intervals, not only p-values.

Success for the strong mechanism claim requires both primary contrasts to be positive after correction. A separability gain without improved behavioral outcomes supports a representation observation, not an improved safety-helpfulness boundary.

## 7. Dimensionality analysis

Treat dimensionality as secondary mechanistic evidence, not as a substitute for held-out separability.

For token-level hidden-state matrices:

- center states consistently;
- compare true reasoning with shuffled/neutral controls using identical token counts;
- use deterministic token subsampling when lengths differ;
- report numerical rank with a fixed tolerance, entropy effective rank, and stable rank;
- measure reasoning-state energy orthogonal to the query/guidance subspace;
- report principal-angle or subspace-overlap diagnostics;
- test whether new-subspace energy predicts held-out probe and behavioral improvement.

Raw rank growth from a longer trace is not evidence for useful dimensional expansion.

## 8. Behavioral linkage

Generate final responses for Direct and true IA on all 3,343 pairs and three seeds. Score them using the already defined shared refusal calibration and both guard models. Run shuffled-analysis IA and empty-analysis IA behavioral responses first on the 100-pair pilot; extend them to all 3,343 pairs only if the measured compute budget permits. Representation-only shuffled/empty replays do not require another stage-1 generation.

For each condition, report:

- Intent Error (lower is better);
- Harmful Outcome (lower is better);
- Behavioral Boundary Margin (higher is better);
- refusal and harmful-score distributions by class;
- per-pair association between representation delta and behavioral delta.

Do not claim mediation from correlation alone. The true-versus-shuffled randomized-content contrast supplies the cleaner causal test.

## 9. Execution gates

### Gate 0: code and data readiness

- fix the checkpoint confound and probe/statistical issues found in strict review;
- add data-audit, source-deduplication, split-manifest, control-construction, model-match, and partial-output tests;
- make Slurm scripts derive or require the project root and respect the cluster's four-CPU limit for CPU-only jobs;
- pass unit and integration tests locally and on the server.

### Gate 1: post-fix pilot

Run 100 pairs from 100 distinct harmful sources with three seeds. Proceed only if:

- every pair/seed/condition is complete;
- exact token reconstruction checks pass;
- no sample is truncated;
- true and shuffled analysis lengths satisfy the frozen tolerance;
- label-permutation performance is near chance;
- rerunning the probe reproduces the same metrics;
- all manifests, logs, and environment records are present.

### Gate 2: frozen 3,343-pair main experiment

Freeze a preregistration-style JSON/YAML manifest containing hypotheses, contrasts, all 3,343 pair IDs, harmful-source-grouped folds, prompts, model/method commits, decoding settings, layer/pooling, probe settings, weighting estimands, and analysis code commit. Then run the full dataset without changing those choices.

### Gate 3: source-reuse robustness

Using the same generated outputs, report the official pair-weighted result, a source-balanced estimate, and the frozen canonical 1,243-pair sensitivity result. Differences among them quantify the impact of harmful-source reuse.

## 10. Follow-up experiments

Only after Experiment 1 is frozen:

1. **Native reasoning:** use a switchable Qwen3 model in a 2 x 2 design: thinking off/on crossed with IA guidance absent/present. Keep model weights, prompts, decoding, data, and thinking budget fixed. Test the interaction, not just four unrelated means. Start with the frozen 100-pair pilot and expand only after budgeting.
2. **Inference-time methods:** Direct, IA, Goal Prioritization, and SAGE under the same representation and behavioral protocol.
3. **Training-based methods:** add one method at a time through a reproducible model/adaptor revision, then rerun the frozen paired evaluation.
4. **Cross-domain generalization:** train probes on safety-topic groups and evaluate on held-out groups after category metadata is added and audited.

Qwen3 is suitable for the native-reasoning factorial because its technical report explicitly defines switchable thinking and non-thinking modes. Llama-3.1 remains the non-native-reasoning control; prompted IA must not be described as native reasoning.

## 11. Compute and artifact plan

The account snapshot after the smoke runs was 1.16 points used and 998.84 available. Do not extrapolate that number directly to the full factorial.

- Use one A800/A100 80GB GPU per generation or extraction shard.
- Prefer A800/A100 PCIe at 16 points/hour over A100 SXM at 20 points/hour unless measured throughput justifies the premium.
- Do not use the current BF16 extraction configuration on V100 without a separately validated FP16 configuration.
- Shard jobs by 100 pairs and limit job-array concurrency to one or two GPUs.
- Run the 100-pair, three-seed pilot first and use its measured tokens/second, elapsed time, and points to project the 3,343-pair run.
- Reserve at least 150 points as an account safety buffer. Do not submit the full run if generation, extraction, and required scoring are projected to exceed 700 additional points; reduce scope only through a new frozen, source-stratified sampling decision made before inspecting outcomes.
- Use CPU Slurm jobs for probes and statistics; request at most four CPUs per task on this cluster.

Every run directory must preserve:

- code commit and dirty-state record;
- model identifier/revision and model-file manifest;
- external method repository commit;
- exact prompts and chat-template hash;
- generation parameters and seeds;
- split manifest and its checksum;
- environment freeze, CUDA/GPU information, submitted Slurm script, job IDs, stdout, and stderr;
- generation rows, representation manifest/shards, per-example predictions, metrics, and final report.

Use a new immutable `RUN_TAG` for every experiment definition. Never overwrite or combine artifacts produced under different prompts, scoring rules, calibration data, or code commits.

## References

- Jiang et al., [WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models](https://arxiv.org/abs/2406.18510).
- Zou et al., [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405).
- Yang et al., [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388).
- Sahoo et al., [Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States](https://arxiv.org/abs/2606.02907).
