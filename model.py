import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class TripletDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = f"subject: {row['subject']} predicate: {row['predicate']} object: {row['object']}"
        target_text = row['text']
        
        # Tokenize inputs and targets
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

class TripletToSentenceModel:
    def __init__(self, model_name='t5-small'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
    def prepare_dataset(self, train_data, val_data):
        """Prepare datasets for training"""
        train_dataset = TripletDataset(train_data, self.tokenizer)
        val_dataset = TripletDataset(val_data, self.tokenizer)
        return train_dataset, val_dataset
        
    def train(self, train_dataset, val_dataset, output_dir='./t5_triplet_model'):
        """Train the model"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss'
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        trainer.train()
        
        # Save the model and tokenizer
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
    def generate_sentence(self, subject, predicate, obj):
        """Generate a sentence from a triplet"""
        input_text = f"subject: {subject} predicate: {predicate} object: {obj}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate
        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=50,
            num_beams=4,
            length_penalty=0.6,
            early_stopping=True
        )
        
        # Decode and return the generated sentence
        generated_sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_sentence
        
    def evaluate_batch(self, test_data, batch_size=32):
        """Evaluate model on a batch of test data"""
        self.model.eval()
        generated_sentences = []
        reference_sentences = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(test_data), batch_size)):
                batch = test_data[i:i+batch_size]
                for _, row in batch.iterrows():
                    generated = self.generate_sentence(row['subject'], row['predicate'], row['object'])
                    generated_sentences.append(generated)
                    reference_sentences.append(row['text'])
                    
        return generated_sentences, reference_sentences 