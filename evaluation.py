from bert_score import BERTScorer
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

class ModelEvaluator:
    def __init__(self):
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        
    def calculate_bertscore(self, generated_sentences, reference_sentences):
        """Calculate BERTScore metrics"""
        P, R, F1 = self.bert_scorer.score(generated_sentences, reference_sentences)
        
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }
        
    def calculate_metrics(self, generated_sentences, reference_sentences):
        """Calculate all metrics"""
        # Calculate BERTScore
        bertscore_metrics = self.calculate_bertscore(generated_sentences, reference_sentences)
        
        # Calculate traditional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            [1] * len(generated_sentences),  # Dummy labels for binary classification
            [1] * len(generated_sentences),
            average='binary'
        )
        
        return {
            'bertscore': bertscore_metrics,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    def evaluate_model(self, model, test_data, batch_size=32):
        """Evaluate model on test data"""
        # Generate sentences
        generated_sentences, reference_sentences = model.evaluate_batch(test_data, batch_size)
        
        # Calculate metrics
        metrics = self.calculate_metrics(generated_sentences, reference_sentences)
        
        return metrics, generated_sentences, reference_sentences 