"""Load (subject, predicate, object) -> sentence pairs from REBEL or a local file."""
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ['subject', 'predicate', 'object', 'text']

# The original dataset repo uses a loading script, which recent `datasets` versions
# no longer run; the Hub's automatic Parquet conversion of the same data is used instead.
REBEL_PARQUET = ('hf://datasets/Babelscape/rebel-dataset@refs%2Fconvert%2Fparquet/'
                 'REBEL/{split}/*.parquet')


def format_triplet(subject, predicate, obj):
    """Model input for one triplet. Used for training and generation alike."""
    return f"subject: {subject} predicate: {predicate} object: {obj}"


def parse_rebel_triplets(linearized):
    """Parse REBEL's linearised triplets into (subject, predicate, object) tuples.

    Format: ``<triplet> head <subj> tail <obj> relation [<subj> tail <obj> relation ...]``
    """
    triplets = []
    for chunk in linearized.split('<triplet>'):
        parts = chunk.split('<subj>')
        head = parts[0].strip()
        for tail_rel in parts[1:]:
            if '<obj>' not in tail_rel:
                continue
            tail, relation = (s.strip() for s in tail_rel.split('<obj>', 1))
            if head and tail and relation:
                triplets.append((head, relation, tail))
    return triplets


_SPACE_BEFORE = re.compile(r'\s+([,.;:!?%)\]])')
_SPACE_AFTER = re.compile(r'([(\[])\s+')


def detokenize(text):
    """Undo REBEL's tokenised spacing, e.g. 'Paris , France ( 2001 ) .' -> 'Paris, France (2001).'"""
    text = _SPACE_BEFORE.sub(r'\1', text.strip())
    text = _SPACE_AFTER.sub(r'\1', text)
    return re.sub(r'\s{2,}', ' ', text)


def rebel_rows(records, single_triplet=True):
    """Turn REBEL records ({'context', 'triplets'}) into rows with REQUIRED_COLUMNS."""
    rows = []
    for record in records:
        triplets = parse_rebel_triplets(record.get('triplets') or '')
        if not triplets or (single_triplet and len(triplets) != 1):
            continue
        subject, predicate, obj = triplets[0]
        rows.append({'subject': subject, 'predicate': predicate, 'object': obj,
                     'text': detokenize(record.get('context') or '')})
    return rows


class DataPreprocessor:
    def __init__(self, max_samples=None, single_triplet=True, val_fraction=0.2, seed=42):
        self.max_samples = max_samples
        self.single_triplet = single_triplet
        self.val_fraction = val_fraction
        self.seed = seed
        self.data = None
        self.train_data = None
        self.val_data = None

    def load_rebel_dataset(self, split='train'):
        """Stream REBEL from the Hugging Face Hub and keep up to ``max_samples`` usable rows.

        With ``single_triplet`` only sentences that express exactly one triplet are kept,
        so the target sentence matches the input.
        """
        try:
            from datasets import load_dataset
            stream = load_dataset('parquet', data_files={split: REBEL_PARQUET.format(split=split)},
                                  split=split, streaming=True)
            rows = []
            for record in stream:
                rows.extend(rebel_rows([record], self.single_triplet))
                if self.max_samples and len(rows) >= self.max_samples:
                    break
        except Exception as exc:
            logger.error('Error loading REBEL: %s', exc)
            return False
        self.data = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
        logger.info('Loaded %d REBEL rows from the %s split', len(self.data), split)
        return True

    def load_local_dataset(self, path):
        """Load a .csv, .json or .jsonl file with subject, predicate, object and text columns."""
        path = Path(path)
        try:
            if path.suffix == '.csv':
                df = pd.read_csv(path)
            elif path.suffix in ('.jsonl', '.json'):
                df = pd.read_json(path, lines=path.suffix == '.jsonl')
            else:
                logger.error('Unsupported data file type: %s', path.suffix)
                return False
        except Exception as exc:
            logger.error('Error reading %s: %s', path, exc)
            return False
        if self.max_samples:
            df = df.head(self.max_samples)
        self.data = df
        return True

    def clean_data(self):
        """Validate columns, drop empty/duplicate rows and split into train/validation."""
        if self.data is None:
            logger.error('Dataset not loaded. Please load the dataset first.')
            return False

        missing = [col for col in REQUIRED_COLUMNS if col not in self.data.columns]
        if missing:
            logger.error('Missing required columns in dataset: %s', ', '.join(missing))
            return False

        df = self.data[REQUIRED_COLUMNS].dropna().astype(str)
        df = df.apply(lambda col: col.str.strip())
        df = df[(df != '').all(axis=1)].drop_duplicates().reset_index(drop=True)
        if len(df) < 2:
            logger.error('Need at least 2 usable rows, got %d', len(df))
            return False

        shuffled = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        n_val = max(1, int(round(len(shuffled) * self.val_fraction)))
        self.val_data = shuffled.iloc[:n_val].reset_index(drop=True)
        self.train_data = shuffled.iloc[n_val:].reset_index(drop=True)
        return True

    def get_processed_data(self):
        """Return the processed train and validation DataFrames."""
        return self.train_data, self.val_data

    def format_triplet(self, row):
        """Format a single row into the model's input/target pair."""
        return {
            'input': format_triplet(row['subject'], row['predicate'], row['object']),
            'target': row['text'],
        }
