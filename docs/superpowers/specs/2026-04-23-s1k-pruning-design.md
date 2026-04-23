# S1K Pruning Study — Extreme SFT Data Efficiency

**Date:** 2026-04-23
**Status:** Design approved (pending user review)

## Question

For SFT on Qwen2.5-32B-Instruct with the s1K reasoning corpus, at what N does
a *curated* subset fail to match the full 1000-sample distillation? And at
equal N, does a skill-coverage-based selection rule outperform uniform random?

## Motivation

Extreme data-efficiency results in recent RL work (Alibaba polymath-RL, Wang
1-shot RLVR) show that a handful of well-chosen examples can match or beat
thousands. We want the SFT analog for the s1K distillation recipe: a scaling
curve that pins the "pain point" of aggressive pruning, and a head-to-head of
two selection rules so we can attribute any gap to *what we keep* rather than
*how much we keep*.

## Scope

### In

- Two pruning strategies: uniform random, and Alibaba-inspired salient-skill
  greedy set-cover.
- Three reduced sizes: N ∈ {500, 250, 100}, plus the existing N=1000 baseline.
- Same Qwen2.5-32B-Instruct SFT pipeline used for the full-s1K baseline.
- Evaluation on MATH500 + AIME24 + GPQA-Diamond via existing `src/s1/eval.py`.

### Out (deferred)

- Smaller sizes (N < 100) — extend later if Phase-1 shows the knee is below 100.
- Multi-seed runs / error bars — single fixed-seed runs for now.
- Other strategy families (longest-trace, domain-stratified, LIMR-learnability,
  embedding clusters) — explicitly rejected upstream to keep the study focused.
- LR / warmup / sequence-length re-tuning — keep existing pipeline unchanged.
- AIME-2024 leakage filtering in the training pool — existing pipeline behavior
  preserved; flag in results if a run looks suspiciously strong on AIME24.
- Smaller-model proxy (7B / 14B) sweep — stay on 32B throughout.

## Run matrix

| N    | Strategy       | Num epochs | Opt. steps | # SFT runs |
| ---- | -------------- | ---------- | ---------- | ---------- |
| 1000 | baseline       | 5          | ~315       | 0 (reuse)  |
| 500  | random         | 10         | ~310       | 1          |
| 500  | skill_cover    | 10         | ~310       | 1          |
| 250  | random         | 20         | ~310       | 1          |
| 250  | skill_cover    | 20         | ~310       | 1          |
| 100  | random         | 50         | ~310       | 1          |
| 100  | skill_cover    | 50         | ~310       | 1          |

**Total new SFT runs: 6.** Each run runs the same ~315 optimizer-step budget as
the full-s1K 5-epoch baseline. Epoch counts chosen so that
`num_epochs × ceil(N / batch_size=16) ≈ 315`.

## Compute normalization

Fix the **optimizer-step budget** at the baseline value (~315 steps on the full
1000-sample × 5-epoch run). Scale `--num-epochs` per N so every cell gets the
same number of gradient updates regardless of dataset size. This makes the pain
point a measurement of *data-signal sufficiency*, not of training duration.

- Warmup stays at 5% of total steps (~16 steps across all cells) — no
  schedule surgery.
- Cosine decay over the full ~315 steps.
- Small-N cells therefore run many epochs on few examples (N=100 → 50 epochs),
  in the same memorization-through-repetition regime as the grokking run and
  the 1-shot RLVR literature. This is intentional.

## Strategies

### 1. Random-uniform

Draw N rows uniformly without replacement from the 1000-sample pool, with a
fixed Python seed of **42**. Deterministic given the seed.

Purpose: the null. Any skill-coverage gain must clear this bar to count.

### 2. Salient-skill-coverage (Alibaba-inspired)

A two-step pipeline: a one-off LLM-judge tagging pass, then a deterministic
greedy set-cover selection.

#### Skill taxonomy (fixed, 23 skills)

Math (14): `algebra`, `precalculus`, `geometry`, `number_theory`,
`combinatorics`, `probability`, `calculus`, `linear_algebra`, `analysis`,
`functional_equations`, `inequalities`, `trigonometry`, `diophantine`,
`olympiad_techniques`.

Science (8): `mechanics`, `electromagnetism`, `thermodynamics`,
`quantum_mechanics`, `optics`, `chemistry`, `biology`, `astronomy`.

Other (1): `crossword_wordplay`.

Chosen to roughly span the `source_type` distribution in s1K. Frozen before
tagging so selection is not biased by what the judge "likes to emit".

#### Tagging pass

Script: `scripts/tag_s1k_skills.py`.

Input: `data/s1K/s1k.parquet` (1000 rows).
For each row, send `(question, thinking_trajectories[0])` and the skill list
to Gemini 2.5 Flash-Lite via the existing Gemini OpenAI-compatible API setup
used for leakage detection. Prompt the judge to return 1–3 skills that the
problem/trace actively exercises (not incidental mentions). JSON output,
strict schema.

Output: `data/s1K/s1k_skills.parquet` with columns:
`index, question, cot_type, source_type, skills` (list[str]), `skill_count`.

Reuse across all skill-cover cells. Re-running with the same model + prompt
should produce nearly identical tags; the file is kept under `data/s1K/`
alongside the other derived parquets. Not committed to git (standard practice
for generated data); reproducible from the tagging script.

#### Greedy set-cover selection

Script: `scripts/prune_s1k.py --strategy skill_cover --n N --seed 42`.

1. Load `s1k_skills.parquet`. Let `S_i ⊆ Skills` be sample *i*'s tag set.
2. `selected = []`, `covered = ∅`.
3. While `len(selected) < N`:
   - Candidate scoring: `gain_i = |S_i \ covered|`, then `skill_count_i`, then
     `-index_i` (smallest index wins the final tiebreak — deterministic).
   - Pick the highest-ranked unselected sample. Add to `selected`, update
     `covered ← covered ∪ S_i`.
4. When `covered == Skills`, subsequent picks have `gain_i = 0`, so the
   skill-count tiebreak takes over and continues picking multi-skill samples.

Fully deterministic. `--seed` is accepted but unused for this strategy; we
keep it for CLI uniformity with the random strategy.

## Evaluation

Use `src/s1/eval.py` unchanged. For each of the 6 new SFT checkpoints plus
the reused N=1000 baseline, evaluate on:

- **MATH500** — 500 problems, numeric / boxed answers.
- **AIME24** — 30 problems, integer 000–999.
- **GPQA-Diamond** — 198 PhD-level multiple-choice science questions.

Report pass@1 per benchmark. Aggregate into a single "avg" column for
quick-look tables.

Expected signature of a win: skill-coverage should gain over random mostly on
GPQA (since random on a math-heavy s1K tends to under-sample science),
possibly also on AIME24. MATH500 is expected to be saturated and
less-sensitive.

## Artifacts

- `data/s1K/s1k_skills.parquet` — produced once by the tagging script, reused across all skill-cover cells.
- `data/s1K/pruned/<strategy>_n<N>_seed42.parquet` — one per cell.
- `checkpoints/s1_prune_<strategy>_n<N>/` — SFT outputs.
- `eval_results/s1_prune_<strategy>_n<N>.json` — per-run eval.
- `results/pruning_sweep.json` — aggregated scoreboard (strategy, N,
  math500, aime24, gpqa, avg).

## New code

- `scripts/tag_s1k_skills.py` — Gemini-judge skill tagger.
- `scripts/prune_s1k.py --strategy {random,skill_cover} --n N --seed SEED
  --input data/s1K/s1k.parquet --output data/s1K/pruned/<...>.parquet`.
  After writing the pruned parquet, re-run `prepare_s1k_sft.py` formatting
  inline (or call into it) to emit the matching `*_sft.parquet`.
- `scripts/submit_prune_sweep.slurm` — SLURM array (or serial) launcher that
  iterates over the 6 cells, calling `run_sft_s1k.sh` with the per-cell
  `TRAIN_FILE`, `NUM_EPOCHS`, and `OUTPUT_DIR`.
- No changes to `sft_trainer.py` or `run_sft_s1k.sh` itself — parameters are
  passed via existing environment variables.

## Risks

- **Skill tagging noise.** A mis-assigned skill biases the set-cover. Partial
  mitigation: constrain the judge to a fixed taxonomy (done) and spot-check
  ~20 tagged samples before committing the parquet.
- **Small-N overfitting to s1K idiosyncrasies.** With 50 epochs on 100
  samples, the model memorizes phrasings. Eval on held-out benchmarks should
  still penalize this.
- **AIME leakage.** s1K's AIME 1983–2024 slice may include 2024 problems. Not
  filtering in-scope per user decision; if N=100 skill-cover suddenly spikes
  on AIME24, check overlap post-hoc.
- **Single-seed runs.** Variance at small N may swallow the skill-cov vs
  random gap. Deferred to follow-up if first pass is inconclusive.

## Success criteria

1. The scaling curve for random on (MATH500, AIME24, GPQA, avg) vs N is
   monotone-ish and identifies where performance drops relative to the N=1000
   baseline. "Pain point" = the smallest N where avg score is within the
   uncertainty range of full-s1K (heuristic, not a hard test given no seeds).
2. At each N ∈ {500, 250, 100}, skill-coverage either (a) beats random on the
   average or (b) beats random on GPQA specifically. Either outcome is
   publishable; neither outcome is also publishable as a negative result.
3. Full sweep runnable end-to-end from a single SLURM submission.
