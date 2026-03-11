# One-Shot RLVR Metric Evaluation

## Motivation

In one-shot RLVR (Reinforcement Learning with Verifiable Rewards), a language model is trained on a single example over many steps using algorithms like GRPO. Surprisingly, some individual examples can push validation accuracy to levels comparable to training on datasets of 1000+ examples — while others barely help at all.

We want to understand **what makes an example good** by computing various metrics on the base model *before* any RL training, and checking whether those metrics predict actual RLVR performance.

## Benchmark Data

We draw on two papers that have systematically tested individual examples and reported their downstream RLVR performance:

### Wang et al. — "RL for Reasoning with One Training Example"
- **Model**: Qwen2.5-Math-1.5B + GRPO
- **Evaluation**: MATH500 accuracy (primary), AIME24, average over 6 benchmarks
- **Examples**: 14 tested (π₁ through π₁₂₀₉), with MATH500 scores ranging from 74.4% (π₁₃) down to 45.0% (π₁₂₀₈)
- **Notable findings**: π₁₂₀₇ has a wrong label (model can never get reward), π₁₂₀₈ is too difficult (near-zero pass rate)
- **Data**: `wang_examples_benchmark.csv`, parquet files in `One-Shot-RLVR/data/train/one_shot_rlvr/`

### Li et al. — "One Sample to Rule Them All"
- **Model**: Qwen2.5-7B-base + GRPO
- **Evaluation**: Average accuracy across 8 domains (Math, Physics, Chemistry, Biology, Science, Engineering, CS, Others)
- **Examples**: 14 tested (7 natural polymath + 6 synthetic specialist + π₁), with averages ranging from 30.8% (synthetic prime) down to 23.8% (natural geometry)
- **Data**: `li_examples_benchmark.csv`, parquet files in `polymath-learning/data/`

## Metrics to Evaluate

We implement and test metrics from three categories. Full descriptions are in `rlvr_example_selection_metrics.md`.

### Output-Based Metrics (from sampling rollouts)
| # | Metric | What it measures |
|---|--------|-----------------|
| 1 | Pass rate @k | Basic difficulty / solvability |
| 2 | Answer distribution entropy | Diversity of model outputs |
| 3 | Unique wrong answers × pass rate | Error surface combined with difficulty |
| 4 | Suffix-conditioned pass rate variance | Sensitivity to prompt framing |
| 5 | Salient skill count | Problem complexity (via strong LLM) |

### Activation-Based Metrics (from model internals)
| # | Metric | What it measures |
|---|--------|-----------------|
| 6 | Per-token entropy profile | Internal uncertainty map over reasoning chain |
| 7 | Hidden state variance across rollouts | Computational path diversity |
| 8 | Attention pattern diversity | Reasoning strategy diversity |
| 9 | Gradient norm of correct answer | Distance from current weights to correct answer |

### Composite Metrics
| # | Metric | Formula |
|---|--------|---------|
| 10 | Error Surface Score | unique_wrong_answers × pass_rate × num_entropy_spikes |
| 11 | Problem Richness Score | salient_skill_count × answer_entropy × f(pass_rate) |

## Approach

1. **Extract examples** from the parquet files (using `data_selection.py` for Wang examples, directly from category folders for Li examples)
2. **Run base model inference** — sample each example multiple times (64–128 rollouts) to get outputs, logits, and hidden states
3. **Compute metrics** for each example
4. **Compare metric rankings** against the known RLVR performance rankings from the benchmark CSVs
5. **Identify** which metrics (or combinations) best predict whether an example will be effective for one-shot RLVR

## Repository Structure

```
one_shot_metrics/
├── README.md                          # This file
├── rlvr_example_selection_metrics.md  # Detailed metric descriptions
├── rlvr_benchmark_examples.md         # Full benchmark data from both papers
├── wang_examples_benchmark.csv        # Wang et al. results (ground truth)
├── li_examples_benchmark.csv          # Li et al. results (ground truth)
├── One-Shot-RLVR/                     # Wang et al. data and scripts
│   └── data/
│       ├── data_selection.py          # Extract specific π examples from parquet
│       ├── data_selection.sh          # Example usage of data_selection.py
│       ├── acc_step_500.json          # Per-example accuracy curves (for ranking/selection)
│       ├── deepscaler_dataset.py      # Dataset preparation script
│       ├── train/one_shot_rlvr/       # Training parquet files (π₁, π₂, π₁₃, etc.)
│       └── test/math500.parquet       # MATH500 test set
└── polymath-learning/                 # Li et al. data
    └── data/
        ├── test.parquet               # Test set
        ├── pi_1/                      # π₁ example
        ├── math_full/                 # MATH 8k baseline
        ├── limr_full/                 # LIMR 1k baseline
        ├── polymath_natural_*/        # Natural polymath samples (by category)
        └── polymath_synthetic_*/      # Synthetic specialist samples (by category)
```
