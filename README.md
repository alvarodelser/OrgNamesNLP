# OrgNamesNLP: Linguistic Patterns in European Public Organization Names

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green)

This repository contains the official implementation of the comparative study: **"Linguistic Patterns in European Public Organization Names"**. It addresses the challenge of classifying public sector organizations across multiple European languages using *only* their official names, under low-context and low-resource assumptions.

## Abstract

This work addresses the challenge of classifying public sector organizations across multiple European languages using only their official names, a critical step for entity disambiguation in knowledge graph population. We employ ontology-based knowledge extraction to evaluate three Natural Language Processing approaches: **rule-based keyword extraction**, **zero-shot Natural Language Inference**, and **embedding-based semantic similarity** — under low-context, low-resource assumptions. Large Language Models are integrated across all three techniques.

Our methodology systematically evaluates multilingual preprocessing, various state-of-the-art models, different supervision regimes, classification structures, and parameter optimization. We conduct a detailed evaluation across three specific domains (healthcare, administration, education) spanning all European Union countries, analyzing performance in relation to lexical structure and class balance.

**Key Findings:**
- ⚡ **Rule-based methods (TF-IDF keyword selection)** are surprisingly effective in multilingual scenarios when limited training data is available.
- 🎯 **Natural Language Inference (NLI) models** offer competitive zero-shot performance but exhibit vulnerabilities to unbalanced class distributions.
- 🌐 **Embedding-based semantic similarity methods** provide the most consistent generalization across languages, showcasing evidence of class coherence in vector space.

This work highlights the feasibility of ontology-guided model training from short texts, offering valid approaches for entity disambiguation in formal knowledge representation systems.

---

## 🏗️ Repository Structure

The project is structured modularly for reproducibility and ease of use:

- **`orgpackage/`**: Core library for the project. Includes modules for:
  - Data ingestion, processing, and sampling (`dataprocesser.py`).
  - Rule-based keyword extraction and classification (`ruleclassifier.py`).
  - Embeddings generation and orchestration (`clusterembedder.py`, `finetuner.py`).
  - Evaluation and permutation testing (`evaluator.py`, `tester.py`).
  - Plotting and metric visualization (`plotter.py`).
- **`scripts/`**: Executable scripts for running different experiment groups.
  - Experiment execution: `run_nli_tests.py`, `run_finetuned_experiments.py`, `ollamaexps.py`.
  - Statistical testing: `run_embedding_permutation_analysis.py`, `build_cd_diagram_data.py`.
  - Aggregation tools: `build_correctness_tables.py`, `report_mean_f1.py`.
- **`data/`**: Datasets and configuration files required for the experiments.
- **`notebooks/`**: Jupyter notebooks for exploratory data analysis, plotting, and inspection of results.
- **`prompts/`**: LLM prompts used for data augmentation, extraction, or zero-shot classification variations.
- **`fasttext_models/`, `keywords/`, `results/`**: Model artifacts, keyword banks, outputs, and generated CSV/JSON results.

## ⚙️ Installation

To set up the project locally, ensure you have Python 3.9+ installed and run the following:

```bash
# Clone the repository
git clone https://github.com/alvarodelser/OrgNamesNLP.git
cd OrgNamesNLP

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install required dependencies
pip install -r requirements.txt
```

> **Note:** The `transformers` and `sentence-transformers` libraries require PyTorch. Ensure that you have a PyTorch version compatible with your hardware (CUDA/MPS/CPU) for optimal inference performance. 

## 🚀 Usage

Experiments are built to be self-contained and run via the scripts or notebooks using orgpackage modules and functions. Resulting experiments are stored in a table results/experiments.csv and metrics, accuracy measures, and F1 scores will be automatically stored in the `results/` folder, which can later be aggregated or visualized.

## Evaluation & Domains

The evaluation scales comprehensively over **all EU member countries**, operating natively on the official organization names across three principal semantic domains:
1. 🏥 **Healthcare:** Hospitals, clinics, medical boards.
2. 🏛️ **Administration:** Ministries, local councils, federal registries.
3. 🎓 **Education:** Universities, research institutes, early education centers.

Analysis extensively reviews whether short-text contextualization relies more significantly on common linguistic roots across lexical families or on language-specific keyword overlap.

## License & Citation

If you use this repository or our findings in your own research or project, please cite the associated paper:

```bibtex
@article{delser_orgnamesnlp,
  title={Linguistic Patterns in European Public Organization Names: A Comparative Study},
  author={del Ser, Álvaro and others},
  journal={Semantic Web Journal (Under Review)},
  year={2024},
  url={https://github.com/alvarodelser/OrgNamesNLP}
}
```

*(Keywords: Ontology Extraction, Natural Language Processing, Multilingual, Low-Context, Low-Resource, Entity Disambiguation)*
