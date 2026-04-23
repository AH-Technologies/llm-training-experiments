# S1K Pruning Study — Extreme SFT Data Efficiency

**Date:** 2026-04-23
**Status:** Design approved (pending user review)

## Question

For SFT on Qwen2.5-32B-Instruct with the s1K reasoning corpus, at what N does
a *curated* subset fail to match the full 1000-sample distillation? And at
equal N, does the Alibaba polymath-RL salient-skill-abundance selection rule
— ported unchanged from RL to SFT distillation — outperform uniform random?

## Motivation

Extreme data-efficiency results in recent RL work (Alibaba polymath-RL, Wang
1-shot RLVR) show that a handful of well-chosen examples can match or beat
thousands. We want the SFT analog for the s1K distillation recipe: a scaling
curve that pins the "pain point" of aggressive pruning, and a head-to-head of
two selection rules so we can attribute any gap to *what we keep* rather than
*how much we keep*.

## Scope

### In

- Two pruning strategies: uniform random, and the paper-native salient-skill
  abundance rule from Alibaba polymath-RL (§4, App. C).
- Three reduced sizes: N ∈ {500, 250, 100}, plus the existing N=1000 baseline.
- Same Qwen2.5-32B-Instruct SFT pipeline used for the full-s1K baseline.
- Evaluation on MATH500 + AIME24 + GPQA-Diamond via existing `src/s1/eval.py`.

### Out (deferred)

- Smaller sizes (N < 100) — extend later if Phase-1 shows the knee is below 100.
- Multi-seed runs / error bars — single fixed-seed runs for now.
- Other strategy families (longest-trace, domain-stratified, LIMR-learnability,
  embedding clusters, greedy set-cover) — explicitly rejected upstream to keep
  the study focused.
- LR / warmup / sequence-length re-tuning — keep existing pipeline unchanged.
- AIME-2024 leakage filtering in the training pool — existing pipeline behavior
  preserved; flag in results if a run looks suspiciously strong on AIME24.
- Smaller-model proxy (7B / 14B) sweep — stay on 32B throughout.

## Run matrix

| N    | Strategy          | Num epochs | Opt. steps | # SFT runs |
| ---- | ----------------- | ---------- | ---------- | ---------- |
| 1000 | baseline          | 5          | ~315       | 0 (reuse)  |
| 500  | random            | 10         | ~310       | 1          |
| 500  | skill_abundance   | 10         | ~310       | 1          |
| 250  | random            | 20         | ~310       | 1          |
| 250  | skill_abundance   | 20         | ~310       | 1          |
| 100  | random            | 50         | ~310       | 1          |
| 100  | skill_abundance   | 50         | ~310       | 1          |

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

Purpose: the null. Any skill-abundance gain must clear this bar to count.

### 2. Salient-skill-abundance (Alibaba polymath-RL §4, App. C)

Faithful port of the paper's method. Two steps: a one-off LLM-judge skill
identification pass using the paper's exact prompt and per-category protocol,
then a deterministic ranking by total skill count.

#### Skill categories (paper-native, 5)

Fixed to the paper's math-only taxonomy, with the paper's own merge rule
applied:

- `algebra` (merged from the paper's *Prealgebra* + *Algebra* +
  *Intermediate Algebra* at identification time, per paper §5 — "we employ
  *Algebra* to include salient skills from *Prealgebra*, *Algebra* and
  *Intermediate Algebra* to eliminate their large overlaps")
- `number_theory`
- `geometry`
- `precalculus`
- `probability`

**Math-only is intentional.** The paper's finding is that salient *math*
skills (especially algebra and precalculus, per Fig. 3) are what drive
cross-domain reasoning transfer — including to physics, chemistry, biology.
A physics or biology problem can still be tagged with high algebra/precalculus
skill counts if solving it draws on those math skills. Science and crossword
problems in s1K that genuinely exercise zero math skills will rank low and
likely be dropped at small N; this is a known implication of the paper's rule,
not a bug.

#### Tagging pass

Script: `scripts/tag_s1k_skills.py`.

Input: `data/s1K/s1k.parquet` (1000 rows).

For each row, run **5 separate skill-identification calls** — one per
category. Each call uses the paper's Table 8 prompt template verbatim with
`[CATEGORY]` and `[QUESTION]` substituted:

```
Here is a reasoning problem, and your job is to identify the concepts and
skills in the scope of [CATEGORY] that are related to solve the problem.
Please separate the concepts or skills with :, and if there is no skills or
concepts identified, please answer with None. Please put your answer within
<answer></answer>.
For example: compute derivatives is the skill in precalculus.
Question: [QUESTION]
```

The judge emits open-vocabulary skill phrases within the scope of that
category, colon-separated, inside `<answer>...</answer>`. "None" → empty
list for that category.

**Judge model: Gemini 2.5 Flash-Lite** via the existing Gemini
OpenAI-compatible API infrastructure already used for leakage detection.
(The paper uses Qwen2.5-72B-instruct; we substitute our available API judge.
Prompt, per-category protocol, merge rule, and downstream selection rule
stay paper-exact — the judge model is the only deviation, and is called out
explicitly in the Risks section.)

**Inference settings:** temperature 0, single response per call, for
reproducibility. (The paper does not specify skill-ID sampling
hyperparameters.)

**Parsing:** extract the string between `<answer>` and `</answer>`; split on
`:`; strip whitespace; lowercase; drop tokens that are literally `None` or
empty. Result: one list of skill phrases per (row, category).

Cost: 1000 rows × 5 categories = **5000 Gemini Flash-Lite calls**, one-off.
Budget-trivial.

Output: `data/s1K/s1k_skills.parquet` with columns:
- `index` (int)
- `question`, `cot_type`, `source_type` (carried through for audit)
- `skills_algebra`, `skills_number_theory`, `skills_geometry`,
  `skills_precalculus`, `skills_probability` (each `list[string]`)
- `skill_count` (int, the sum of the five list lengths — the selection key)

Reuse across all skill-abundance cells. Kept under `data/s1K/`; not committed
to git; reproducible from the tagging script.

#### Skill-abundance selection

Script: `scripts/prune_s1k.py --strategy skill_abundance --n N`.

1. Load `s1k_skills.parquet`.
2. Sort rows by `(skill_count DESC, index ASC)`. Index serves as the
   deterministic tiebreak — no randomness.
3. Take the top N rows.

Matches the paper's specialist-sample selection (§4: "we then select the
problems with the most specialized skills"). Fully deterministic.

The `--seed` flag is accepted for CLI uniformity with the random strategy
but ignored here.

## Evaluation

Use `src/s1/eval.py` unchanged. For each of the 6 new SFT checkpoints plus
the reused N=1000 baseline, evaluate on:

- **MATH500** — 500 problems, numeric / boxed answers.
- **AIME24** — 30 problems, integer 000–999.
- **GPQA-Diamond** — 198 PhD-level multiple-choice science questions.

Report pass@1 per benchmark. Aggregate into a single "avg" column for
quick-look tables.

Expected signature of a win (per paper §6): skill-abundance should gain over
random strongest on AIME24 and MATH500 (dense math-skill demands match
abundance-selected samples), with possible cross-domain transfer lift on
GPQA-Diamond via algebra/precalculus skills (paper Fig. 3). A GPQA gap in
*favor of random* would actually be consistent with the paper's rule
downselecting away science-framed problems — worth calling out either way.

## Artifacts

- `data/s1K/s1k_skills.parquet` — produced once by the tagging script, reused across all skill-abundance cells.
- `data/s1K/pruned/<strategy>_n<N>_seed42.parquet` — one per cell.
- `checkpoints/s1_prune_<strategy>_n<N>/` — SFT outputs.
- `eval_results/s1_prune_<strategy>_n<N>.json` — per-run eval.
- `results/pruning_sweep.json` — aggregated scoreboard (strategy, N,
  math500, aime24, gpqa, avg).

## New code

- `scripts/tag_s1k_skills.py` — Gemini-judge skill tagger. Runs 5 per-category
  prompts per row over all 1000 rows. Emits `data/s1K/s1k_skills.parquet`.
  Uses paper's Table 8 prompt template verbatim.
- `scripts/prune_s1k.py --strategy {random,skill_abundance} --n N --seed SEED
  --input data/s1K/s1k.parquet --output data/s1K/pruned/<...>.parquet`.
  After writing the pruned parquet, re-run `prepare_s1k_sft.py` formatting
  inline (or call into it) to emit the matching `*_sft.parquet`.
- `scripts/submit_prune_sweep.slurm` — SLURM array (or serial) launcher that
  iterates over the 6 cells, calling `run_sft_s1k.sh` with the per-cell
  `TRAIN_FILE`, `NUM_EPOCHS`, and `OUTPUT_DIR`.
- No changes to `sft_trainer.py` or `run_sft_s1k.sh` itself — parameters are
  passed via existing environment variables.

## Risks

- **Judge-model deviation from paper.** Paper uses Qwen2.5-72B-instruct; we
  use Gemini 2.5 Flash-Lite via API. A weaker/different judge could inflate
  or deflate skill counts systematically, shifting which samples rank top.
  Mitigation: spot-check ~20 tagged samples across categories before
  committing; consider a one-category sanity re-tag with a stronger judge if
  results look anomalous.
- **Math-only taxonomy drops s1K's science slice.** By design — the paper's
  rule does not tag physics/chem/bio/crossword skills. At N=100 the selected
  set will skew math-heavy. If GPQA-Diamond performance collapses under
  skill-abundance selection, that is the expected failure mode of the paper's
  rule applied to a mixed-domain corpus, not a bug.
- **Small-N overfitting to s1K idiosyncrasies.** With 50 epochs on 100
  samples, the model memorizes phrasings. Eval on held-out benchmarks should
  still penalize this.
- **AIME leakage.** s1K's AIME 1983–2024 slice may include 2024 problems. Not
  filtering in-scope per user decision; if N=100 skill-abundance suddenly
  spikes on AIME24, check overlap post-hoc.
- **Single-seed runs.** Variance at small N may swallow the skill-abundance
  vs random gap. Deferred to follow-up if first pass is inconclusive.

## Success criteria

1. The scaling curve for random on (MATH500, AIME24, GPQA, avg) vs N is
   monotone-ish and identifies where performance drops relative to the N=1000
   baseline. "Pain point" = the smallest N where avg score is within the
   uncertainty range of full-s1K (heuristic, not a hard test given no seeds).
2. At each N ∈ {500, 250, 100}, compare skill-abundance to random on
   per-benchmark pass@1 and on the 3-benchmark average. A gain on MATH500
   and/or AIME24 would corroborate the paper's rule. A loss on GPQA-Diamond
   would be a negative-result finding worth reporting (indicates the
   math-only selection rule trades cross-domain breadth for math-skill
   density). Any of these outcomes is informative.
3. Full sweep runnable end-to-end from a single SLURM submission.
