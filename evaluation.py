"""Metrics for generated sentences: BLEU, ROUGE, token-overlap P/R/F1 and (optionally) BERTScore."""
import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r'\w+', re.UNICODE)


def token_overlap(generated, reference):
    """Bag-of-words precision/recall/F1 between two sentences (case-insensitive)."""
    gen = Counter(_TOKEN.findall(generated.lower()))
    ref = Counter(_TOKEN.findall(reference.lower()))
    overlap = sum((gen & ref).values())
    if not gen or not ref or not overlap:
        return 0.0, 0.0, 0.0
    precision = overlap / sum(gen.values())
    recall = overlap / sum(ref.values())
    return precision, recall, 2 * precision * recall / (precision + recall)


class ModelEvaluator:
    def __init__(self, use_bertscore=True):
        # BERTScore downloads roberta-large (~1.4 GB) on first use, so it is loaded lazily.
        self.use_bertscore = use_bertscore
        self._bert_scorer = None

    def calculate_bertscore(self, generated_sentences, reference_sentences):
        if self._bert_scorer is None:
            from bert_score import BERTScorer
            self._bert_scorer = BERTScorer(lang='en', rescale_with_baseline=True)
        P, R, F1 = self._bert_scorer.score(generated_sentences, reference_sentences)
        return {'precision': P.mean().item(), 'recall': R.mean().item(), 'f1': F1.mean().item()}

    @staticmethod
    def calculate_bleu(generated_sentences, reference_sentences):
        import sacrebleu
        return sacrebleu.corpus_bleu(generated_sentences, [reference_sentences]).score / 100

    @staticmethod
    def calculate_rouge(generated_sentences, reference_sentences):
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        totals = Counter()
        for gen, ref in zip(generated_sentences, reference_sentences):
            for name, score in scorer.score(ref, gen).items():
                totals[name] += score.fmeasure
        n = max(len(generated_sentences), 1)
        return {name: totals[name] / n for name in ('rouge1', 'rouge2', 'rougeL')}

    def calculate_metrics(self, generated_sentences, reference_sentences):
        """Corpus-level metrics, all in [0, 1]."""
        if not generated_sentences:
            raise ValueError('No sentences to evaluate')
        overlaps = [token_overlap(g, r) for g, r in zip(generated_sentences, reference_sentences)]
        n = len(overlaps)
        metrics = {
            'bleu': self.calculate_bleu(generated_sentences, reference_sentences),
            'rouge': self.calculate_rouge(generated_sentences, reference_sentences),
            # Token-overlap scores replace the old placeholder metrics, which compared
            # dummy labels and were always 1.0.
            'precision': sum(o[0] for o in overlaps) / n,
            'recall': sum(o[1] for o in overlaps) / n,
            'f1': sum(o[2] for o in overlaps) / n,
            'bertscore': None,
        }
        if self.use_bertscore:
            metrics['bertscore'] = self.calculate_bertscore(generated_sentences, reference_sentences)
        return metrics

    def evaluate_model(self, model, test_data, batch_size=32):
        """Generate sentences for ``test_data`` and score them."""
        generated, references = model.evaluate_batch(test_data, batch_size)
        return self.calculate_metrics(generated, references), generated, references
