# Triplet-to-Sentence Generation

A deep learning project that generates natural language sentences from knowledge triplets using T5 models.

## Overview

This project implements a system that converts structured knowledge triplets (subject-predicate-object) into natural language sentences. It uses a fine-tuned T5 model and includes components for data processing, model training, sentence generation, and evaluation.

## Features

- Data processing for REBEL and WebNLG datasets
- T5 model fine-tuning for triplet-to-sentence generation
- Enhanced graph processing for context-aware generation
- Comprehensive evaluation metrics (BLEU, ROUGE, semantic similarity)
- Grammar and fact checking
- Hybrid generation approach combining templates and neural models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/triplet-sentence-generator.git
cd triplet-sentence-generator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Process the dataset:
```bash
python scripts/rebel_processor.py
```

2. Train the model:
```bash
python run.py
```

3. Generate sentences:
```bash
python main.py
```

4. Evaluate results:
```bash
python scripts/evaluate_sentences.py --input data/test_triplets.json
```

## Project Structure

```
triplet-sentence-generator/
├── data/                   # Data directory
├── models/                 # Model implementations
│   ├── enhanced_graph_processor.py
│   ├── hybrid_sentence_generator.py
│   ├── t5_generator.py
│   └── t5_model.py
├── scripts/               # Processing and utility scripts
│   ├── rebel_processor.py
│   ├── webnlg_processor.py
│   ├── sentence_evaluator.py
│   └── evaluate_sentences.py
├── validators/            # Validation components
│   └── sentence_validator.py
├── main.py               # Main execution script
├── run.py               # Training wrapper
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Evaluation Metrics

The system uses multiple metrics to evaluate generated sentences:
- BLEU Score
- ROUGE Score (ROUGE-1, ROUGE-2, ROUGE-L)
- Semantic Similarity
- Grammar Check
- Fact Checking

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- REBEL dataset: [Babelscape/rebel-dataset](https://huggingface.co/datasets/Babelscape/rebel-dataset)
- WebNLG dataset
- T5 model: [Google Research](https://github.com/google-research/text-to-text-transfer-transformer) 