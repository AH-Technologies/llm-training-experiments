#!/usr/bin/env python3
"""Fetch wandb test scores and plot EMA-smoothed accuracy curves."""
import wandb
import matplotlib.pyplot as plt
import numpy as np

api = wandb.Api()
entity = "alexauren-ntnu"
project = "attention-illuminates-qwen3"

run_names = [
    "rhythm_runA_qwen3_base_v6",
    "rhythm_runC_qwen3_rediscovery_v6",
    "rhythm_runB_qwen3_illuminates_v6",
]

metrics = [
    "val-core/simplerl/math500/acc/mean@1",
    "val-core/aime_2025/acc/mean@1",
    "val-core/amc_2023/acc/mean@1",
]
metric_labels = ["MATH500", "AIME 2025", "AMC 2023"]


def display_name(name):
    name = name.replace("rhythm_run", "").replace("_qwen3_", " ").replace("_", " ")
    return name


def ema_ratchet(values, alpha=0.3):
    out = np.zeros_like(values)
    out[0] = values[0]
    for t in range(1, len(values)):
        out[t] = alpha * max(values[t], out[t - 1]) + (1 - alpha) * out[t - 1]
    return out


print("Fetching runs...")
all_runs = api.runs("{}/{}".format(entity, project))
run_map = {}
for r in all_runs:
    if r.name in run_names:
        run_map[r.name] = r
        print("  Found: {} (state={}, steps={})".format(r.name, r.state, r.summary.get("_step", "?")))

run_data = {}
for name in run_names:
    if name not in run_map:
        print("  MISSING: {}".format(name))
        continue
    r = run_map[name]
    hist = r.history(keys=["_step"] + metrics, samples=5000)
    rows = []
    for _, row in hist.iterrows():
        step = row.get("_step")
        if step is None:
            continue
        vals = {m: row.get(m) for m in metrics}
        if any(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in vals.values()):
            rows.append({"step": int(step), **vals})
    run_data[name] = rows
    print("  {} : {} data points".format(name, len(rows)))

alpha = 0.3
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(run_names)))

for ax, metric, label in zip(axes, metrics, metric_labels):
    for i, name in enumerate(run_names):
        if name not in run_data or not run_data[name]:
            continue
        rows = run_data[name]
        max_step = 280
        steps = [r["step"] for r in rows if r["step"] <= max_step and r.get(metric) is not None and not (isinstance(r[metric], float) and np.isnan(r[metric]))]
        vals = [r[metric] for r in rows if r["step"] <= max_step and r.get(metric) is not None and not (isinstance(r[metric], float) and np.isnan(r[metric]))]
        if not vals:
            continue
        smoothed = ema_ratchet(np.array(vals, dtype=float), alpha=alpha)
        ax.plot(steps, smoothed, label=display_name(name), color=colors[i], linewidth=2)
        ax.plot(steps, vals, color=colors[i], alpha=0.15, linewidth=0.8)

    ax.set_title(label, fontsize=13)
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(run_names), fontsize=10, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("EMA-ratchet smoothed test accuracy - v6 runs (alpha={})".format(alpha), fontsize=14, y=1.01)
fig.tight_layout()

out_path = "attention_sparks_thinking/logs/test_accuracy_ema_v6.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print("\nSaved to {}".format(out_path))
