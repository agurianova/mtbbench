# MTBBench extension (Hancock pilot)

Extensions on the Hancock track: compare **Llama 3.1 8B** (`unsloth/Llama-3.1-8B-Instruct`) and **Meditron 8B** (`epfl-llm/meditron3-8b`) on text-only questions, measure calibration, and test SEER/TCGA population hints for prognosis.

## Code changes (vs upstream)

- **Models** — `neurips25/utils/load_model.py`: vLLM text eval for Llama and Meditron.
- **Agent** — `neurips25/models/agent.py`: `[ANSWER]` / `[CONFIDENCE]` tags, log-prob + self-reported confidence, optional cohort context.
- **Population** — staging + TCGA/SEER lookup (`staging.py`, `survival_lookup.py`, `data/hnc_survival_reference.json`); naive rule: favorable if cohort rate > 50% + 10 pp.
- **CLI** — `--no_population_stats`, `--skip_choice_prob_scoring`.
- **Analysis** — `neurips25/analysis/`: question types, text-only filter, ECE/AUROC, prognosis follow/disagree.

Runs use plain `DoctorAgent`.

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

Use `--skip_choice_prob_scoring` if GPU memory is tight.

## Figures

- Notebook: `notebooks/mtbbench_extension_figures.ipynb`
- Script: `python notebooks/mtbbench_figures.py`
- Output: `notebooks/figures/` (PNG only)

Generated files: `medical_llm_text_only_overall`, `cal_reliability_logprobs`, `cal_auroc_panel`, `cohort_usage_panel` (accuracy when following vs disagreeing with cohort rule), `cohort_case_example_116` (PIL diagram). Log dir names: `notebooks/log_paths.py`.

## Results on current logs

**No population:** Llama 58.3%, Meditron 64.1% (text-only, n=156).

**With population (text-only):** Llama 66.7%, Meditron 67.9%.

**Prognosis only, with injection:** Llama 63.5%, Meditron 69.2%; naive population-rule baseline 67.3%.

Calibration plots use the no-population logs; cohort plots use `*_with-population` and compare to no-population prognosis runs for the ablation chart.
