# Simulating a Committee's Group Decision With LLM Agents (HDCA)

**K J Somaiya College of Engineering, Vidyavihar, Mumbai - 400 077**  
**Department of Information Technology | M.Tech AI&DS (2026-27) | NLP Mini Project (IA1)**

## Group Members
- **Rafay Ahmed**
- **Jainam Shah**

## Problem Statement
Single-agent Large Language Models (LLMs) suffer from cognitive degeneration, factual hallucinations, overconfidence, and brittle decision boundaries. Existing multi-agent committees using naive Unweighted Majority Voting (UMV) or unconstrained debate often collapse into artificial consensus and sycophancy. This project implements **Heterogeneous Deliberative Committee Aggregation (HDCA)**, a 3-round multi-agent governance framework combining complementary persona conditioning (`Cautious`, `Neutral`, `Lenient`), decoupled Round 1 private initialization, Round 2 peer critique, and Round 3 preference aggregation.

## Dataset
- **MMLU-Pro** (`TIGER-Lab/MMLU-Pro`): 12,032 expert-level multiple-choice questions across 14 disciplines (10 options per question), partitioned 20-20-60 (Train / Validation calibration set $D_{cal}$ / Test) with 15% adversarial role perturbation.
- **Domain Case Studies** (`data/sample_cases.json`):
  1. **Loan Approval Decision** (`gemini-3.5-flash-lite`, `temperature=0.2`)
  2. **Rubric-Based Exam Grading** (`gemini-3.5-flash`, `temperature=0.2`)

## Repository Layout
```text
├── README.md
├── requirements.txt
├── paper_data_code_availability.tex
├── data/
│   ├── download_mmlu_pro.py
│   └── sample_cases.json
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── hdca.py
│   ├── baselines.py
│   ├── evaluate.py
│   └── generate_plots.py
├── notebooks/
│   └── nlp_ia.ipynb
├── checkpoints/
│   ├── loan_case_checkpoint.json
│   └── exam_case_checkpoint.json
├── logs/
│   ├── loan_simulation.log
│   ├── exam_simulation.log
│   └── benchmark_evaluation.log
└── results/
    ├── accuracy_comparison.png
    ├── precision_comparison.png
    ├── recall_comparison.png
    ├── f1_comparison.png
    ├── multi_metric_comparison.png
    ├── exam_score_shift_trajectory.png
    ├── metrics_comparison.csv
    └── metrics_summary.json
```

## Setup and Run Steps
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Gemini API key (required only for live LLM calls)
export GEMINI_API_KEY="your_api_key_here"

# 3. Download and split MMLU-Pro (seed=42)
python data/download_mmlu_pro.py

# 4. Reproduce evaluation tables and publication plots (offline, seed=42)
python -m src.evaluate
python -m src.generate_plots
```

## Results Summary

### MMLU-Pro Benchmark Comparison
| Metric | Solo Baseline | UMV | Proposed HDCA | Gain over UMV (pp) |
| :--- | :---: | :---: | :---: | :---: |
| Accuracy | 81.2% | 85.4% | **92.5%** | +7.1 |
| Precision | 79.5% | 82.1% | **91.8%** | +9.7 |
| Recall | 82.1% | 80.5% | **90.4%** | +9.9 |
| F1-Score | 80.8% | 81.3% | **91.1%** | +9.8 |

### Executed Notebook Case Studies
- **Loan Approval**: Solo Agent = `APPROVE` | HDCA Committee = `APPROVE` (2 APPROVE, 1 REJECT; Consensus Rate = `66.67%`; Stance Shifts = `0`).
- **Exam Grading**: Solo Agent = `2.0/10` | HDCA Committee = `2.33/10` (Agent A: `1.0 -> 1.0`, Agent B: `3.0 -> 1.0`, Agent C: `6.0 -> 5.0`; Consensus Rate = `66.67%`; Stance Shifts = `2` / `66.7%`).
