# Benchmark: All Examples Tested in Both Papers with RLVR Performance

## Paper 1: Wang et al. — "RL for Reasoning with One Training Example"

### Model: Qwen2.5-Math-1.5B + GRPO (default)

Base model performance: MATH500 = 36.0%, Avg 6 benchmarks = 17.6%

| Example | Category | Prompt (shortened) | Ground Truth | MATH500 (%) | AIME24 (%) | Avg 6 bench (%) | Best Step | Notes |
|---------|----------|-------------------|--------------|-------------|------------|-----------------|-----------|-------|
| π₁ | Algebra | Wind pressure on sail, P=kAV³, find velocity | 12.8 | 74.0 | 16.7 | 35.0 | 1540 | Base model 6.3% pass@128, diverse wrong answers |
| π₂ | Number Theory | Common positive divisors of 9240 and 13860 | 24 | 70.6 | 17.1 | 33.5 | 320 | |
| π₃ | Counting & Prob. | Committee of 5 from 10 with leaders | 7560 | — | — | — | — | In {π₁...π₁₆} set only |
| π₄ | Number Theory | Three integers from {1,2,4,8,16,20} product 80 | 25 | 65.6 | 17.1 | — | 80 | Fast saturation, low final acc |
| π₅ | Counting & Prob. | 4×4 array with row/col/quadrant constraints | 288 | — | — | — | — | In {π₁...π₁₆} set only |
| π₆ | Geometry | 3×1×1 prism, 16 points, distances = √2 | 32 | — | — | — | — | In {π₁...π₁₆} set only |
| π₇ | Intermediate Alg. | Recurrence u_{k+1}=2u_k-2u_k², find k for convergence | 10 | 64.0 | 12.1 | — | 580 | |
| π₈ | Number Theory | Sums of 3 distinct from {2,7,12,17,22,27,32} | 13 | — | — | — | — | In {π₁...π₁₆} set only |
| π₉ | Counting & Prob. | 4 boys 3 girls no adjacent same gender | 144 | — | — | — | — | In {π₁...π₁₆} set only |
| π₁₀ | Number Theory | Ten-digit numbers with ≥2 identical digits | 8996734080 | — | — | — | — | In {π₁...π₁₆} set only |
| π₁₁ | Number Theory | Pairs (a,b) in 1..42 with a⁹≡b⁷ mod 43 | 42 | 64.0 | 13.3 | — | 20 | Very fast saturation |
| π₁₂ | Physics/Algebra | Two springs in series, work to stretch 10cm | 20 | — | — | — | — | In {π₁...π₁₆} set only |
| π₁₃ | Geometry | Circle through 3 points + line intersects circle | 4/3 | 74.4 | 17.1 | 35.7 | 2000 | Base model 21.9% pass@128 |
| π₁₄ | Counting & Prob. | 7 cards, arrangements leaving 6 in order | 74 | — | — | — | — | In {π₁...π₁₆} set only |
| π₁₅ | Geometry | Geoboard quadrilateral area (ASY code) | 22.5 | — | — | — | — | In {π₁...π₁₆} set only |
| π₁₆ | Algebra | p+q+r=26, 1/p+1/q+1/r+360/pqr=1, find pqr | 576 | 67.0 | 14.6 | — | 600 | |
| π₁₇ | Counting & Prob. | 45 students, 39 shuttlecock, 28 basketball, find all 3 | 22 | 67.2 | 13.3 | — | 220 | |
| π₆₀₅ | Precalculus | Vectors m⊥n, find f(x) monotonic intervals + triangle | (√3-1)/4 | 71.8 | 14.6 | — | 1040 | Medium historical variance |
| π₆₀₆ | Number Theory | Zeros at end of s(1)·s(2)·...·s(100) | 19 | 64.4 | 14.2 | — | 460 | Medium historical variance |
| π₁₂₀₁ | Geometry | Quadrilateral angles ∠P=3∠Q=4∠R=6∠S | 206 | 71.4 | 16.3 | 33.7 | 1120 | Low historical variance |
| π₁₂₀₇ | Geometry | Paper folding B/A ratio | 4/5 (WRONG, correct=2/3) | 54.0 | 9.6 | — | 100 | Wrong label, model never gets reward |
| π₁₂₀₈ | Counting & Prob. | Quadratic f(x)=ax²-4bx+1, probability questions | 961/1280 | 45.0 | 8.8 | — | 240 | Too difficult, almost no reward |
| π₁₂₀₉ | Precalculus | f(x)=xe^x, 2023rd derivative, y-intercept | -2023/2024 | 72.2 | 17.5 | 33.5 | 1220 | Low historical variance |
| π₁' (simplified) | Computation | Calculate ∛2048 | 12.8 | 65.4 | 9.6 | 30.0 | — | Only the crux step, much worse |

### Model: Qwen2.5-Math-7B + GRPO

Base: MATH500 = 51.0%, Avg = 22.4%

| Example | MATH500 (%) | AIME24 (%) | Avg 6 bench (%) |
|---------|-------------|------------|-----------------|
| {π₁} | 79.2 | 23.8 | 40.2 |
| {π₁,π₁₃} | 79.2 | 21.7 | 41.3 |
| {π₁,π₂,π₁₃,π₁₂₀₉} | 78.6 | 22.5 | 42.5 |
| {π₁...π₁₆} | 77.8 | 30.4 | 42.5 |
| Random 16 | 76.0 | 22.1 | 40.2 |
| DSR-sub (1209) | 78.6 | 25.8 | 42.8 |

### Model: Llama-3.2-3B-Instruct + GRPO

Base: MATH500 = 40.8%, Avg = 17.5%

| Example | MATH500 (%) | Avg 6 bench (%) |
|---------|-------------|-----------------|
| {π₁} | 45.8 | 19.0 |
| {π₁,π₁₃} | 49.4 | 21.0 |
| {π₁,π₂,π₁₃,π₁₂₀₉} | 46.4 | 19.8 |
| DSR-sub (1209) | 43.2 | 19.8 |

### Model: Qwen2.5-Math-1.5B + PPO

| Example | MATH500 (%) | Avg 6 bench (%) |
|---------|-------------|-----------------|
| {π₁} | 72.4 | 33.8 |
| DSR-sub (1209) | 72.8 | 35.4 |

### Model: DeepSeek-R1-Distill-Qwen-1.5B + GRPO (32k eval)

Base: MATH500 = 82.9%, Avg = 44.9%

| Example | MATH500 (%) | Avg 6 bench (%) |
|---------|-------------|-----------------|
| {π₁} | 83.9 | 46.3 |
| {π₁,π₂,π₁₃,π₁₂₀₉} | 84.8 | 46.9 |
| {π₁...π₁₆} | 84.5 | 48.3 |
| DSR-sub (1209) | 84.5 | 48.6 |

---

## Paper 2: Li et al. — "One Sample to Rule Them All"

### Model: Qwen2.5-7B-base + GRPO (all results)

Base 0-shot pass@64: Math=20.4, Physics=4.4, Chemistry=4.4, Biology=5.1, Avg=6.4

#### Natural Polymath Samples (1-shot RLVR)

| Sample Category | Prompt (shortened) | Answer | Math | Physics | Chemistry | Biology | Science | Engineering | CS | Others | Avg |
|----------------|-------------------|--------|------|---------|-----------|---------|---------|-------------|-----|--------|-----|
| Geometry | Cylindrical silo, red stripe area | 240 | 15.5 | 9.9 | 10.0 | 55.1 | 11.2 | 16.7 | 37.1 | 35.0 | 23.8 |
| Prealgebra | Semicircular arcs on square perimeter | 4 | 38.0 | 17.4 | 12.2 | 51.7 | 15.1 | 16.5 | 49.5 | 33.5 | 29.2 |
| Algebra | 100-gon P1→P2→P3 x-coord sum | 2009 | 37.3 | 17.4 | 13.7 | 51.7 | 12.1 | 15.6 | 43.3 | 30.9 | 27.7 |
| Intermediate Algebra | a/b+b/c+c/a=7, b/a+c/b+a/c=9, find sum of cubes | 157 | 36.3 | 19.1 | 13.1 | 50.0 | 13.9 | 17.5 | 42.3 | 31.1 | 27.9 |
| Number Theory | Freshman class mod 23 and mod 21 | 413 | 37.7 | 16.9 | 12.4 | 49.2 | 13.4 | 17.8 | 42.3 | 32.2 | 27.7 |
| Precalculus | System of equations, find xz/y² | 10 | 38.0 | 18.4 | 13.7 | 50.0 | 16.0 | 19.7 | 43.3 | 31.0 | 28.8 |
| Probability | Flatville bicycle plates, add 2 letters | 40 | 38.8 | 19.9 | 11.5 | 46.6 | 14.7 | 16.4 | 41.2 | 31.4 | 27.6 |

#### Synthetic Specialist Samples (1-shot RLVR)

| Sample Category | Answer | Math | Physics | Chemistry | Biology | Science | Engineering | CS | Others | Avg |
|----------------|--------|------|---------|-----------|---------|---------|-------------|-----|--------|-----|
| Geometry (cell membrane cube) | 800 | 35.4 | 15.0 | 11.5 | 31.1 | 36.1 | 52.5 | 13.2 | 11.0 | 25.7 |
| Algebra (DNA + catalase + ions) | 20 | 37.3 | 16.9 | 12.6 | 31.5 | 41.2 | 52.5 | 18.6 | 13.9 | 28.1 |
| Number Theory (drug half-life) | 10 | 38.4 | 18.2 | 12.0 | 32.1 | 36.1 | 47.5 | 18.6 | 13.8 | 27.1 |
| Precalculus (waste discharge optimization) | 10 | 37.1 | 20.3 | 15.3 | 32.9 | 44.3 | 48.3 | 20.8 | 16.5 | 29.4 |
| Probability (plasmid ³²P decay) | 8 | 37.1 | 16.7 | 13.9 | 30.1 | 46.4 | 50.0 | 19.7 | 10.8 | 28.1 |
| **Prime** (DNA + H-bonds + photons) | **2** | **38.3** | **20.6** | **15.7** | **54.2** | **15.6** | **20.8** | **48.5** | **32.4** | **30.8** |

#### π₁ on Qwen2.5-7B-base (for comparison)

| Sample | Math | Physics | Chemistry | Biology | Science | Engineering | CS | Others | Avg |
|--------|------|---------|-----------|---------|---------|-------------|-----|--------|-----|
| π₁ | 35.5 | 14.3 | 11.3 | 28.4 | 35.1 | 44.1 | 13.8 | 10.4 | 24.1 |

#### Comprehensive Learning baselines

| Dataset | Size | Math | Physics | Chemistry | Biology | Science | Engineering | CS | Others | Avg |
|---------|------|------|---------|-----------|---------|---------|-------------|-----|--------|-----|
| MATH | 8000 | 37.2 | 12.8 | 10.0 | 31.4 | 6.5 | 8.6 | 25.8 | 23.4 | 19.5 |
| LIMR | 1000 | 38.0 | 11.6 | 11.8 | 48.3 | 10.0 | 13.4 | 35.1 | 31.5 | 25.0 |

### Model: Qwen2.5-14B-base + GRPO

| Sample | Math | Physics | Chemistry | Biology | Science | Engineering | CS | Others | Avg |
|--------|------|---------|-----------|---------|---------|-------------|-----|--------|-----|
| 0-shot | 37.7 | 26.2 | 22.2 | 28.1 | 41.2 | 39.0 | 20.8 | 14.3 | 28.7 |
| MATH 8k | 42.7 | 26.4 | 20.5 | 44.7 | 49.5 | 64.4 | 22.3 | 15.6 | 35.8 |
| π₁ | 40.4 | 27.6 | 20.0 | 39.4 | 51.5 | 57.6 | 22.1 | 17.1 | 34.5 |
| Prime | 44.0 | 32.7 | 22.7 | 42.3 | 56.7 | 58.5 | 31.0 | 20.6 | 38.6 |

---

## Ranking Summary for Metric Validation

### Wang et al. examples ranked by MATH500 on Qwen2.5-Math-1.5B (your primary benchmark):

1. π₁₃ = 74.4%
2. π₁ = 74.0%
3. π₁₂₀₉ = 72.2%
4. π₆₀₅ = 71.8%
5. π₁₂₀₁ = 71.4%
6. π₂ = 70.6%
7. π₁₇ = 67.2%
8. π₁₆ = 67.0%
9. π₄ = 65.6%
10. π₆₀₆ = 64.4%
11. π₇ = 64.0%
12. π₁₁ = 64.0%
13. π₁₂₀₇ = 54.0% (wrong label)
14. π₁₂₀₈ = 45.0% (too difficult)

### Li et al. examples ranked by Avg across domains on Qwen2.5-7B-base:

1. Synthetic Prime = 30.8%
2. Synthetic Precalculus = 29.4%
3. Natural Prealgebra = 29.2%
4. Natural Precalculus = 28.8%
5. Synthetic Algebra = 28.1%
6. Synthetic Probability = 28.1%
7. Natural Intermediate Algebra = 27.9%
8. Natural Algebra = 27.7%
9. Natural Number Theory = 27.7%
10. Natural Probability = 27.6%
11. Synthetic Number Theory = 27.1%
12. Synthetic Geometry = 25.7%
13. π₁ = 24.1%
14. Natural Geometry = 23.8%

### Known base model pass rates (from the papers):
- π₁ on Qwen2.5-Math-1.5B: 6.3% pass@128 (label "12.8"), 57.8% output "12.7"/"12.70"
- π₁₃ on Qwen2.5-Math-1.5B: 21.9% pass@128
- π₁₂₀₇: ~0% (wrong label, model can never match)
- π₁₂₀₈: ~0% (too difficult, almost never correct)
