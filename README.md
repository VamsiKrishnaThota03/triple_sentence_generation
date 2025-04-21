# Triplet-to-Sentence Generation using T5

This project implements a sentence generation model based on the Pre-trained Language Model (T5) using triplets. The model takes subject-predicate-object triplets as input and generates natural language sentences.

## Features

- Data preprocessing for the REBEL dataset
- Fine-tuning of T5 model for triplet-to-sentence generation
- Comprehensive evaluation using BERTScore metrics
- Example generation from custom triplets

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `data_preprocessing.py`: Handles loading and preprocessing of the REBEL dataset
- `model.py`: Contains the T5 model implementation and training logic
- `evaluation.py`: Implements evaluation metrics and model assessment
- `main.py`: Main script to run the complete pipeline

## Usage

1. Run the main script:
```bash
python main.py
```

This will:
- Load and preprocess the REBEL dataset
- Train the T5 model
- Evaluate the model's performance
- Generate example sentences from test triplets

## Model Architecture

The model uses the T5 architecture with the following components:
- Input: Formatted triplet (subject, predicate, object)
- Output: Generated natural language sentence
- Training: Fine-tuning on the REBEL dataset
- Evaluation: BERTScore metrics (precision, recall, F1-score)

## Results

The model achieves:
- BERTScore Precision: ~0.66
- BERTScore Recall: ~0.59
- BERTScore F1: ~0.62

## Example Usage

```python
from model import TripletToSentenceModel

# Initialize model
model = TripletToSentenceModel()

# Generate sentence from triplet
sentence = model.generate_sentence(
    subject="Albert Einstein",
    predicate="developed",
    obj="theory of relativity"
)
print(sentence)
```

## Future Improvements

1. Enhanced evaluation using human judgments
2. Additional metrics such as BLEU scores
3. Extension to other datasets
4. Interactive tool development for knowledge graphs
5. Question-answering task integration

## License

This project is licensed under the MIT License - see the LICENSE file for details. 