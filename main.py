from data_preprocessing import DataPreprocessor
from model import TripletToSentenceModel
from evaluation import ModelEvaluator
import os
import torch
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize components
    logger.info("Initializing components...")
    preprocessor = DataPreprocessor()
    model = TripletToSentenceModel()
    evaluator = ModelEvaluator()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    if not preprocessor.load_rebel_dataset():
        logger.error("Failed to load REBEL dataset")
        return
        
    if not preprocessor.clean_data():
        logger.error("Failed to clean dataset")
        return
        
    train_data, val_data = preprocessor.get_processed_data()
    logger.info(f"Processed data: {len(train_data)} training samples, {len(val_data)} validation samples")
    
    # Prepare datasets for training
    logger.info("Preparing datasets for training...")
    train_dataset, val_dataset = model.prepare_dataset(train_data, val_data)
    
    # Train model
    logger.info("Starting model training...")
    model.train(train_dataset, val_dataset)
    logger.info("Model training completed")
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics, generated_sentences, reference_sentences = evaluator.evaluate_model(
        model, val_data, batch_size=32
    )
    
    # Log results
    logger.info("Evaluation Results:")
    logger.info(f"BERTScore Precision: {metrics['bertscore']['precision']:.4f}")
    logger.info(f"BERTScore Recall: {metrics['bertscore']['recall']:.4f}")
    logger.info(f"BERTScore F1: {metrics['bertscore']['f1']:.4f}")
    logger.info(f"Traditional Precision: {metrics['precision']:.4f}")
    logger.info(f"Traditional Recall: {metrics['recall']:.4f}")
    logger.info(f"Traditional F1: {metrics['f1']:.4f}")
    
    # Example generation
    logger.info("\nExample generations:")
    test_triplets = [
        {
            'subject': 'Albert Einstein',
            'predicate': 'developed',
            'object': 'theory of relativity'
        },
        {
            'subject': 'William Shakespeare',
            'predicate': 'wrote',
            'object': 'Hamlet'
        }
    ]
    
    for triplet in test_triplets:
        generated = model.generate_sentence(
            triplet['subject'],
            triplet['predicate'],
            triplet['object']
        )
        logger.info(f"\nInput triplet: {triplet}")
        logger.info(f"Generated sentence: {generated}")

if __name__ == "__main__":
    main() 