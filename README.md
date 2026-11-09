# Natural Language Generation from Knowledge Triplets Using Pre-trained T5 Models

A research project on turning knowledge-graph triplets *(subject, predicate, object)* into fluent English sentences by fine-tuning a T5 encoder–decoder. The repository holds the paper source (LaTeX) and a PyTorch / Hugging Face training and evaluation pipeline.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.30-FFD21E)
![LaTeX](https://img.shields.io/badge/LaTeX-elsarticle-008080?logo=latex&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**Authors:** Vamsi Krishna, Tejeshwar Rao, Ruthvika Talagampala, Nida Afseen  
*PDPM Indian Institute of Information Technology, Design and Manufacturing, Jabalpur, India*

## Abstract

Generating natural language from structured data is central to report generation, question answering and conversational systems. Template-based systems are reliable but repetitive, while purely neural generators are fluent but can drift from the facts they are given. This work studies a sequence-to-sequence approach in which a pre-trained **T5** model is fine-tuned on triplet–sentence pairs derived from the **REBEL** dataset. Each triplet is linearised into a prompt, and the model learns to produce a sentence that expresses the relation. The paper also outlines a sentence-validation stage (grammar, structure, coherence and semantic checks) and an evaluation protocol based on n-gram and embedding-based metrics.

The full write-up is in [`main.tex`](main.tex) (Elsevier `elsarticle`, Procedia format).

## Method

### Problem formulation

Given a triplet $t = (s, p, o)$, generate a sentence $y$ that is grammatical and faithful to $t$. The task is cast as text-to-text generation:

```
input : "subject: <s> predicate: <p> object: <o>"
target: <reference sentence>
```

### Pipeline in this repository

| Stage | File | What it does |
|-------|------|--------------|
| Data loading | `data_preprocessing.py` | Loads `Babelscape/rebel` with 🤗 `datasets`, converts the train split to pandas, drops duplicate and empty rows, checks for `subject` / `predicate` / `object` / `text` columns and makes an 80/20 train/validation split (`random_state=42`) |
| Dataset | `model.py` → `TripletDataset` | Linearises each triplet and tokenises inputs and targets (max length 512, padded and truncated) |
| Model | `model.py` → `TripletToSentenceModel` | `T5ForConditionalGeneration` initialised from `t5-small` (configurable). Runs on CUDA when available. |
| Training | `model.py` → `train()` | Hugging Face `Trainer`: 10 epochs, batch size 64, 500 warm-up steps, weight decay 0.01. Evaluates and checkpoints every epoch, keeps the best model by validation loss and saves it to `./t5_triplet_model` |
| Decoding | `model.py` → `generate_sentence()` | Beam search (4 beams), `max_length=50`, `length_penalty=0.6` |
| Evaluation | `evaluation.py` → `ModelEvaluator` | **BERTScore** precision, recall and F1 (English, baseline-rescaled) between generated and reference sentences |
| Driver | `main.py` | Loads data → trains → evaluates on the validation split → prints example generations (e.g. *(Albert Einstein, developed, theory of relativity)*) |

### Described in the paper (not part of this code)

The manuscript also describes components that are **not** implemented in the Python files here: a LanguageTool-based sentence validator with an improve-and-retry loop, BLEU / ROUGE / grammar-error metrics, a `t5-base` configuration with different hyper-parameters (learning rate 2e-5, batch size 8, 3 epochs), and Apple-Silicon (MPS) support. Treat the code as the baseline training and evaluation pipeline and the paper as the broader design.

## Repository structure

```
triple_sentence_generation/
├── main.tex                 # Paper source (elsarticle / Procedia)
├── main.py                  # End-to-end driver: data → train → evaluate → examples
├── data_preprocessing.py    # REBEL loading, cleaning, train/val split
├── model.py                 # TripletDataset + T5 fine-tuning and generation
├── evaluation.py            # BERTScore-based evaluation
├── requirements.txt
├── LICENSE                  # MIT
└── triplet-sentence-generator   # Git submodule pointer (see note below)
```

> **Note:** `triplet-sentence-generator` is recorded as a submodule commit, but there is no `.gitmodules` file, so it cannot be checked out. The extended implementation it pointed to (validators, BLEU/ROUGE evaluation, WebNLG processing) is not available from this repository.

## Running the code

### Requirements

- Python 3.9+ (pinned versions are in `requirements.txt`: `transformers 4.30.2`, `torch 2.0.1`, `datasets 2.13.1`, `bert-score 0.3.13`, …)
- A CUDA GPU is strongly recommended for fine-tuning. The batch size of 64 at sequence length 512 needs a lot of GPU memory, so lower `per_device_train_batch_size` in `model.py` if needed.
- Internet access on first run, to download `t5-small`, the REBEL dataset and the BERTScore model

### Setup

```bash
git clone https://github.com/VamsiKrishnaThota03/triple_sentence_generation.git
cd triple_sentence_generation

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Train and evaluate

```bash
python main.py
```

The script logs BERTScore P/R/F1 on the validation split and prints generated sentences for the example triplets. The fine-tuned model and tokenizer are saved to `./t5_triplet_model`.

**Data format:** `DataPreprocessor.clean_data()` expects flat `subject`, `predicate`, `object` and `text` columns. The raw `Babelscape/rebel` release stores relations in a different layout (a `context` passage with linearised `triplets`), so you may need to add a conversion step that maps REBEL records to one row per triplet before training.

### Use a trained model

```python
from model import TripletToSentenceModel

m = TripletToSentenceModel(model_name="./t5_triplet_model")
print(m.generate_sentence("William Shakespeare", "wrote", "Hamlet"))
```

## Building the paper

The paper uses Elsevier's `elsarticle` class, which ships with TeX Live and MiKTeX.

```bash
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
# or simply:
latexmk -pdf main.tex
```

`main.tex` references two files that are not in the repository yet:

- `t5_architecture.png`: the architecture figure (compilation stops with an error until you add it or comment out the `\includegraphics` line)
- `references.bib`: the bibliography (BibTeX will warn and the reference list will be empty)

You can also upload `main.tex` (plus the figure and `.bib` file) to Overleaf.

## Limitations

- Training uses only the REBEL train split, re-split 80/20. There is no held-out test split in the code.
- The "traditional precision / recall / F1" printed by `main.py` uses constant dummy labels, so it is always 1.0 and is not a meaningful metric. Use BERTScore instead.
- Results reported in the manuscript come from the extended system described there and cannot be reproduced from this code alone.

## Acknowledgements

- **REBEL** dataset: Huguet Cabot & Navigli, 2021 — [Babelscape/rebel-dataset](https://huggingface.co/datasets/Babelscape/rebel-dataset)
- **T5**: Raffel et al., 2020 — [google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer)
- **BERTScore**: Zhang et al., 2020 — [Tiiiger/bert_score](https://github.com/Tiiiger/bert_score)
- Hugging Face [Transformers](https://github.com/huggingface/transformers) and [Datasets](https://github.com/huggingface/datasets)

## License

Released under the [MIT License](LICENSE).
