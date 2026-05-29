# MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology

**MTBBench** is a benchmark designed to evaluate the reasoning capabilities of multimodal large language models (LLMs) in complex clinical decision-making scenarios. It focuses on two core challenges in oncology: multimodal integration (e.g., pathology, genomics, radiology) and longitudinal reasoning across patient timelines. The benchmark includes agentic tasks requiring interaction with external foundation model-based tools and datasets.

---

## 🛠 Getting Started

To install all required dependencies, simply run:

```bash
bash setup.sh
```

> **Note**: If you want to evaluate the agent on **IHC data**, you will need to additionally clone and install [TRIDENT](https://github.com/mahmoodlab/TRIDENT/tree/main) from source.

---

## ⚙️ Configuration

Before running the benchmark, configure your paths and credentials in:

```
neurips25/configs/base.yaml
```

The config file specifies paths to datasets, tool credentials, and output directories. Below, we provide guidance for acquiring the necessary external datasets.

---

## 📁 Datasets

### HANCOCK (Multimodal Tissue Microarrays)

The **HANCOCK** dataset contains SVS-format tissue micro arrays (TMAs). To prepare it:

1. Follow the original [HANCOCK GitHub repository](https://github.com/ankilab/HANCOCK_MultimodalDataset) to extract tiles and compute cell densities using QuPath.
2. Reproduce our ABMIL training by extracting tumor centers and cell density measurements for **Blocks 1 and 2**.
3. Download the dataset files from the [HANCOCK project page](https://hancock.research.fau.eu/download) to replicate the question curation used in the benchmark.

---

### MSK-CHORD (Longitudinal Genomic Profiles)

The **MSK-CHORD** dataset is available on [cBioPortal](https://www.cbioportal.org/study/summary?id=msk_chord_2024). To use it:

* Download the ZIP archive from the cBioPortal page.
* Extract it and update the dataset path in `base.yaml`.

---

### DrugBank API

To enable the **DrugBank tool** for longitudinal drug lookups:

1. [Register for a DrugBank account](https://www.drugbank.com).
2. Apply for a license to access their API.
3. Download and locally host the dataset following their documentation.
4. Update your API path and credentials in the config file.

---

## 📑 Agent Logs

We provide full logs of agent interactions for all models evaluated in the paper:

* `agent_logs_hancock/`: Multimodal evaluation logs (HANCOCK)
* `agent_logs_msk/`: Longitudinal evaluation logs (MSK-CHORD)

Each log includes all agent–LLM conversations, intermediate reasoning steps, and generated answers.

---

## ▶️ Running the Benchmark

Make sure you have:

* Installed dependencies
* Configured Hugging Face access tokens (for model download)
* Set paths in `base.yaml`

To run an evaluation with `Qwen/Qwen2.5-VL-7B-Instruct` on the HANCOCK dataset:

```bash
python -m neurips25.benchmarks.run_agent_benchmark \
  --doctor_model "Qwen/Qwen2.5-VL-7B-Instruct" \
  --output_dir "./agent_logs_hancock/" \
  --dataset "hancock"
```

---
