# Research Proposal
Summer school research proposal on evolving MTBBench (NeurIPS 2025) Hancock track [🔗 PDF](AIRI_Research_Proposal.pdf)


## Where logs live

Under `agent_logs_hancock/`:

- **`llama31-8b/`**, **`meditron3-8b/`** — no SEER/TCGA in the prompt (`--no_population_stats`). Used for Experiments 1–2 and as the “without injection” arm in Experiment 3.
- **`llama31-8b_with-population/`**, **`meditron3-8b_with-population/`** — prognosis questions get staging, cohort rates, and the population rule. Used for Experiment 3.

Other paper models (Qwen-VL, Gemma, …) keep their original folder names.

**Text-only subset:** 156 of 390 questions (blood + prognosis; image questions excluded).

## Run benchmarks

Configure `neurips25/configs/base.yaml`, then from repo root:

```bash
# Experiments 1 & 2 (no cohort stats)
python -m neurips25.benchmarks.run_agent_benchmark \
  --doctor_model unsloth/Llama-3.1-8B-Instruct \
  --output_dir ./agent_logs_hancock/llama31-8b/ \
  --dataset hancock --no_population_stats

python -m neurips25.benchmarks.run_agent_benchmark \
  --doctor_model epfl-llm/meditron3-8b \
  --output_dir ./agent_logs_hancock/meditron3-8b/ \
  --dataset hancock --no_population_stats

# Experiment 3 (with cohort stats — drop --no_population_stats)
python -m neurips25.benchmarks.run_agent_benchmark \
  --doctor_model unsloth/Llama-3.1-8B-Instruct \
  --output_dir ./agent_logs_hancock/llama31-8b_with-population/ \
  --dataset hancock

python -m neurips25.benchmarks.run_agent_benchmark \
  --doctor_model epfl-llm/meditron3-8b \
  --output_dir ./agent_logs_hancock/meditron3-8b_with-population/ \
  --dataset hancock
```
