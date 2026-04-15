# OrgNamesNLP: Linguistic Patterns in European Public Organization Names

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green)

This repository contains the official implementation of the comparative study: **"Linguistic Patterns in European Public Organization Names"**. It addresses the challenge of classifying public sector organizations across multiple European languages using solely their official names, operating under strict low-context and low-resource constraints.

## Abstract

This work evaluates computational approaches for semantic classification of European public institutions strictly from name strings. Such classification is a foundational task for entity disambiguation, knowledge graph population, and metadata harmonization (Section 1.1: Problem Definition). Employing structured knowledge extraction from Wikidata, the study assesses three Natural Language Processing paradigms: rule-based keyword extraction, zero-shot Natural Language Inference (NLI), and embedding-based semantic similarity. The methodology systematically tests multilingual preprocessing, varying supervision regimes, classification structures, and representation learning techniques. We conduct a comprehensive empirical evaluation spanning healthcare, administration, and education domains across 24 European Union countries and 29 languages.

## Methodology (Section 3)

The computational architecture is structured into three progressive methodological families to handle the brevity and multilingual ambiguity of administrative records (Section 3.2: Classification Experiments Design):

1. **Rule-Based Heuristics** (Section 4.1): Serving as a highly interpretable baseline, this approach utilizes tokenization and algorithmic keyword selection (e.g., TF-IDF and frequency counters). It relies on deterministic regular expressions to capture consistent institutional markers and domain-specific lexicon present in national naming conventions.
2. **Zero-Shot Natural Language Inference (NLI)** (Section 4.2): This method leverages multilingual transformer models (e.g., XLM-RoBERTa, mDeBERTa) by formulating taxonomic classes as entailment hypotheses. It enables categorization without prior labeled instances, relying entirely on the native cross-lingual generalization capabilities of pre-trained encoders.
3. **Embedding-Based Classifiers** (Section 4.3): To overcome the limitations of strictly lexical matches, this strategy maps name strings to dense, high-dimensional vector representations using models such as Multilingual-E5, Mistral, and Qwen. Static embeddings are evaluated using nearest-neighbor similarity, and further finetuned via supervised contrastive loss before training bounded linear classifiers (Logistic Regression and Support Vector Machines).

## Evaluation and Domains (Section 3.1)

The evaluation operates on semantically constrained ground truth datasets constructed via query extraction from Wikidata, targeting functional public utility classes (Section 3.1: Ontology-Guided Data Extraction). The experiments are designed to address differing classification topologies:

- **Medical Domain**: Tests a nested hierarchical structure to differentiate broad healthcare facilities from specialized entities such as university hospitals.
- **Administrative Domain**: Employs a binary classification model identifying local governments as the foundational tier of regional administration.
- **Educational Domain**: Addresses overlapping organizational responsibilities with multi-label classification to resolve facilities acting concurrently as primary and secondary schools.

## Key Findings (Section 4)

- **Lexical Regularities** (Section 4.1: Rule-based Approach): Lightweight rule-based methods (specifically optimized TF-IDF selection) demonstrate substantial performance and computational efficiency. They effectively capture the descriptive and highly standardized naming conventions utilized across European public organizations.
- **Challenges in Zero-Shot NLI** (Section 4.2: Natural Language Inference): While zero-shot architectures generalize reasonably without supervision, their accuracy drastically deteriorates under class imbalance and nested classification structures, struggling to resolve fine-grained threshold distinctions.
- **Semantic Vector Spaces** (Section 4.3: Embedding-based Classification): Embedding methods coupled with Support Vector Machine heads establish the highest predictive performance. Utilizing supervised contrastive fine-tuning drastically enhances class separability (Fisher's Discriminant Ratio) within the latent space, mitigating the intrinsic geometric anisotropy typical of pre-trained models.

## Repository Structure

The project is modularly structured to guarantee reproducibility across the experimental evaluation phases:

- `orgpackage/`: Core modular library containing data ingestion pipelines (`dataprocesser.py`), classifier modules (`ruleclassifier.py`, `finetuner.py`, `clusterembedder.py`), and statistical testing logic.
- `scripts/`: Executable entry points for the experiment families (NLI, embedding generation, permutation testing, and metric aggregations).
- `data/`: Extracted datasets and model configuration files.
- `notebooks/`: Jupyter notebooks dedicated to exploratory lexical data analysis and metric visualization.
- `prompts/`: Specialized prompts for few-shot generative classification experiments.

## Data and Models

- **Dataset Embeddings:** The generated dataset embeddings utilized across the classification experiments are publicly hosted and available on Zenodo: [https://doi.org/10.5281/zenodo.15312407](https://doi.org/10.5281/zenodo.15312407).
- **Trained Models:** The trained classifier artifacts (exported as `.pkl` files) are available directly within this GitHub repository.

## Installation and Usage

To initialize the project environment, require Python 3.9+ and execute the following:

```bash
git clone https://github.com/alvarodelser/OrgNamesNLP.git
cd OrgNamesNLP
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Experiments are self-contained through Python scripts or notebooks leveraging the `orgpackage` modules. Intermediate metrics, experimental variations, and aggregated F1 scores will automatically persist to the `results/` directory structures for downstream statistical analyses.

## License and Citation

If you utilize this methodology or implementation, please cite the underlying academic paper:

```bibtex
@article{delser_orgnamesnlp,
  title={Linguistic Patterns in European Public Organization Names: A Comparative Study},
  author={del Ser, Álvaro and others},
  journal={Semantic Web Journal (Under Review)},
  year={2024},
  url={https://github.com/alvarodelser/OrgNamesNLP}
}
```
