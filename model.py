"""T5 fine-tuning and generation for triplet-to-sentence."""
import logging

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments)

from data_preprocessing import format_triplet

logger = logging.getLogger(__name__)


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class TripletDataset(Dataset):
    """Tokenised (triplet -> sentence) examples. Padding is done per batch by the collator."""

    def __init__(self, data, tokenizer, max_source_length=128, max_target_length=128):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inputs = self.tokenizer(format_triplet(row['subject'], row['predicate'], row['object']),
                                max_length=self.max_source_length, truncation=True)
        labels = self.tokenizer(text_target=row['text'], max_length=self.max_target_length,
                                truncation=True)
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            # The collator pads labels with -100 so padding is ignored by the loss.
            'labels': labels['input_ids'],
        }


class TripletToSentenceModel:
    def __init__(self, model_name='t5-small', max_source_length=128, max_target_length=128):
        self.device = pick_device()
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        logger.info('Loading %s on %s', model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def prepare_dataset(self, train_data, val_data):
        """Wrap the train/validation DataFrames as torch datasets."""
        kwargs = dict(max_source_length=self.max_source_length,
                      max_target_length=self.max_target_length)
        return (TripletDataset(train_data, self.tokenizer, **kwargs),
                TripletDataset(val_data, self.tokenizer, **kwargs))

    def train(self, train_dataset, val_dataset, output_dir='./t5_triplet_model', epochs=10,
              batch_size=64, learning_rate=5e-5, weight_decay=0.01, warmup_steps=500,
              logging_steps=100, max_steps=-1, seed=42):
        """Fine-tune with the Hugging Face Trainer, keeping the checkpoint with the lowest eval loss."""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=f'{output_dir}/logs',
            logging_steps=logging_steps,
            eval_strategy='epoch' if max_steps <= 0 else 'steps',
            save_strategy='epoch' if max_steps <= 0 else 'steps',
            eval_steps=max_steps if max_steps > 0 else None,
            save_steps=max_steps if max_steps > 0 else None,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            report_to=[],
            seed=seed,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=self.model),
        )
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        self.model = trainer.model.to(self.device)
        return trainer

    @torch.no_grad()
    def generate_batch(self, triplets, max_new_tokens=50, num_beams=4):
        """Generate one sentence per (subject, predicate, object) tuple."""
        self.model.eval()
        inputs = self.tokenizer([format_triplet(*t) for t in triplets], padding=True,
                                truncation=True, max_length=self.max_source_length,
                                return_tensors='pt').to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams,
                                      length_penalty=0.6, early_stopping=True)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def generate_sentence(self, subject, predicate, obj):
        """Generate a sentence from a single triplet."""
        return self.generate_batch([(subject, predicate, obj)])[0]

    def evaluate_batch(self, test_data, batch_size=32):
        """Generate sentences for every row of ``test_data``; returns (generated, references)."""
        generated, references = [], []
        for start in tqdm(range(0, len(test_data), batch_size), desc='Generating'):
            batch = test_data.iloc[start:start + batch_size]
            triplets = list(zip(batch['subject'], batch['predicate'], batch['object']))
            generated.extend(self.generate_batch(triplets))
            references.extend(batch['text'].tolist())
        return generated, references
