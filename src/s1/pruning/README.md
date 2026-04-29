# s1.pruning

SFT data-efficiency experiments on s1K. Two execution modes:

- **Screening** (`scripts/submit_prune_screen.slurm`) — fast proxy runs to
  rank candidate strategies. Thirds-based: each strategy splits s1K into
  top/middle/bottom by its score, all three thirds get a short SFT run.
- **Full sweep** (`scripts/submit_prune_sweep.slurm`) — final compute on
  the strategy/N combinations chosen from the screening results. Cosine
  schedule, full step budget, full eval.

This README covers the screening flow.

## Pipeline (per cell)

```
prune (thirds)  →  format for SFT  →  train (3 ep, constant LR)  →
  bf16-cast  →  upload to Hub  →  eval on AMC+AIME25  →  rm -rf local
```

| Step | What | Output |
|---|---|---|
| Prune | `s1.pruning.prune --strategy X --position {top,middle,bottom} --n 333` | `data/s1K/pruned/screen_X_pos.parquet` |
| Format | `scripts/prepare_s1k_sft.py` | `..._sft.parquet` |
| Train | `s1.sft_hf --save-final-only --lr-scheduler-type constant_with_warmup` | `checkpoints/s1_screen_X_pos/` (fp32) |
| Cast + upload | inside `sft_hf.py`: bf16 cast in place, push to `alexauren/s1-pruning/screen/X/pos` | Hub artifact (~65 GiB bf16) |
| Eval | `s1.eval --benchmarks amc aime25` | `eval_results/screen_X_pos.json` |
| Cleanup | `rm -rf` local checkpoint | local disk reclaimed |

After all cells: an aggregation step writes `results/pruning_screen.json` with
per-cell `{strategy, position, amc, aime25, avg}` rows.

## Hyperparameters (held fixed)

- Model: `Qwen/Qwen2.5-32B-Instruct`
- N = **333** per third (hardcoded — `len(s1K) // 3`)
- Epochs = **3** (≈ 63 optimizer steps at batch 16)
- LR = **1e-5 constant** (no decay; `constant_with_warmup` with 5% warmup)
- Sequence length = 16384, per-device batch = 1, global batch = 16
- Weight decay = 1e-4, grad clip = 1.0, optimizer = AdamW (β=0.9/0.95)
- Single final save per cell (`--save-final-only`)

## Validation

The screening eval uses **AMC + AIME 2025** (113 problems total) from
`data/val/{amc_verl,aime_2025_verl}.parquet`. These are loaded by the
`amc` and `aime25` benchmark names in `s1.eval`. MATH500 is intentionally
skipped here because SFT'd 32B models tend to saturate it; AMC + AIME25
discriminate strategies more sharply.

For the full sweep on the surviving strategies, swap to
`--benchmarks math500 aime24 gpqa` (or the full val_combined parquet)
for paper-comparable numbers.

## Hub layout

```
alexauren/s1-pruning/             (public)
├── screen/                       ← screening artifacts
│   ├── response_length/{top,middle,bottom}/
│   ├── skill_count/{top,middle,bottom}/
│   └── ... (one subfolder per strategy × position)
├── random/{1000,500,250,100}/    ← full-sweep cells
└── skill_abundance/{500,250,100}/
```

Each leaf is a self-contained bf16 HF checkpoint
(`config.json` + 29 safetensor shards + tokenizer). Loadable directly:

```python
from vllm import LLM
m = LLM("alexauren/s1-pruning", subfolder="screen/response_length/top")
```

This means re-evaluation later (after editing extractors, swapping
benchmarks, etc.) does not require retraining — just re-run `s1.eval`
against the Hub model.

## Adding a strategy

1. **Register a scoring function** in `strategies.py`:

   ```python
   @register("my_strategy")
   def score_my_strategy(s1k_table, skills_table):
       # return one float per row of s1k_table
       return [...]
   ```

   Score conventions: higher = "more of the property". The thirds split
   sorts descending, so `top` = highest-score third.

2. **Add three lines to `SCREEN_CELLS`** in `submit_prune_screen.slurm`:

   ```bash
   "my_strategy top"
   "my_strategy middle"
   "my_strategy bottom"
   ```

3. Resubmit. Existing cells with `eval_results/screen_*.json` already on
   disk will be skipped — only the new strategy's three cells run.
   Hub-archived prior cells stay untouched and can still be re-evaluated.

The `prune.py --strategy` CLI auto-resolves any name registered in
`strategies.py`. No code changes elsewhere.

## Resumability

- Eval JSON (`eval_results/screen_<strategy>_<position>.json`) is the
  durable completion marker.
- Local checkpoint dirs are deleted after each successful cell, so
  directory presence is **not** a reliable signal.
- If a cell's training succeeds but upload or eval fails, the local
  checkpoint stays on disk; the next submission picks up where it left off
  (training is skipped, only the failed step retries).

## File layout

```
src/s1/pruning/
├── __init__.py
├── README.md           ← this file
├── strategies.py       ← scoring functions (the modular extension point)
├── prune.py            ← random / skill_abundance / thirds-screening selection
└── tag_skills.py       ← one-off skill tagging via Gemini judge

scripts/
├── submit_prune_screen.slurm    ← screening sweep (this README)
├── submit_prune_sweep.slurm     ← full sweep
└── prepare_s1k_sft.py           ← parquet → prompt/response format
```

## Compute budget (rough)

Per cell, sequential on 16 H200s with warm HF cache:

| Step | Time |
|---|---|
| Model load + FSDP init | ~7 min |
| Training (63 steps) | ~10 min |
| save_model + bf16 cast | ~8 min |
| Hub upload (~65 GiB) | ~5 min |
| Eval (AMC + AIME25, 113 problems) | ~5 min |
| **Per cell total** | **~35 min** |

For the initial 6-cell screen (response_length + skill_count, three
positions each): ~3.5 h. Each new strategy adds ~1.7 h.
