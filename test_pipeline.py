"""Fast tests that don't download models: python -m unittest -v test_pipeline"""
import unittest
from pathlib import Path

import pandas as pd

from data_preprocessing import (DataPreprocessor, detokenize, format_triplet,
                                parse_rebel_triplets, rebel_rows)
from evaluation import ModelEvaluator, token_overlap

SAMPLE = Path(__file__).with_name('examples') / 'sample_triplets.csv'


class DataTests(unittest.TestCase):
    def test_parse_rebel_linearisation(self):
        text = ('<triplet> Trinity Peninsula <subj> Graham Land <obj> part of '
                '<subj> Antarctica <obj> continent <triplet> Graham Land <subj> Antarctica <obj> continent')
        self.assertEqual(parse_rebel_triplets(text), [
            ('Trinity Peninsula', 'part of', 'Graham Land'),
            ('Trinity Peninsula', 'continent', 'Antarctica'),
            ('Graham Land', 'continent', 'Antarctica'),
        ])
        self.assertEqual(parse_rebel_triplets(''), [])

    def test_rebel_rows_single_triplet_filter(self):
        records = [
            {'context': 'ITS is located at the Port of Long Beach , California . ',
             'triplets': '<triplet> Port of Long Beach <subj> Long Beach, California <obj> located in'},
            {'context': 'x', 'triplets': '<triplet> A <subj> B <obj> r <subj> C <obj> s'},
        ]
        rows = rebel_rows(records)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['text'], 'ITS is located at the Port of Long Beach, California.')
        self.assertEqual(len(rebel_rows(records, single_triplet=False)), 2)

    def test_detokenize(self):
        self.assertEqual(detokenize('Paris , France ( 2001 ) .'), 'Paris, France (2001).')

    def test_local_csv_and_split(self):
        prep = DataPreprocessor(val_fraction=0.25)
        self.assertTrue(prep.load_local_dataset(SAMPLE))
        self.assertTrue(prep.clean_data())
        train, val = prep.get_processed_data()
        self.assertEqual(len(train) + len(val), len(pd.read_csv(SAMPLE)))
        self.assertEqual(len(set(train['text']) & set(val['text'])), 0)

    def test_missing_columns(self):
        prep = DataPreprocessor()
        prep.data = pd.DataFrame({'subject': ['a'], 'text': ['b']})
        self.assertFalse(prep.clean_data())

    def test_format_triplet(self):
        self.assertEqual(format_triplet('Paris', 'capital of', 'France'),
                         'subject: Paris predicate: capital of object: France')


class MetricTests(unittest.TestCase):
    def test_token_overlap(self):
        self.assertEqual(token_overlap('Paris is nice', 'Paris is nice'), (1.0, 1.0, 1.0))
        self.assertEqual(token_overlap('a b', 'c d'), (0.0, 0.0, 0.0))

    def test_metrics_without_bertscore(self):
        evaluator = ModelEvaluator(use_bertscore=False)
        refs = ['Paris is the capital of France.', 'Berlin is the capital of Germany.']
        perfect = evaluator.calculate_metrics(refs, refs)
        self.assertAlmostEqual(perfect['bleu'], 1.0, places=3)
        self.assertAlmostEqual(perfect['rouge']['rougeL'], 1.0)
        self.assertIsNone(perfect['bertscore'])
        wrong = evaluator.calculate_metrics(['Cats sleep.', 'Dogs bark.'], refs)
        self.assertLess(wrong['f1'], 0.5)  # the old placeholder metrics were always 1.0


if __name__ == '__main__':
    unittest.main()
