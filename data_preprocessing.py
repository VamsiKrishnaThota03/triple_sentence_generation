import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.dataset = None
        self.train_data = None
        self.val_data = None
        
    def load_rebel_dataset(self):
        """Load the REBEL dataset"""
        try:
            # Load the REBEL dataset
            self.dataset = load_dataset("Babelscape/rebel")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
            
    def clean_data(self):
        """Clean and preprocess the dataset"""
        if self.dataset is None:
            print("Dataset not loaded. Please load the dataset first.")
            return False
            
        # Convert to pandas DataFrame for easier manipulation
        df = pd.DataFrame(self.dataset['train'])
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove rows with missing values
        df = df.dropna()
        
        # Ensure all required columns are present
        required_columns = ['subject', 'predicate', 'object', 'text']
        if not all(col in df.columns for col in required_columns):
            print("Missing required columns in dataset")
            return False
            
        # Split into train and validation sets
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        self.train_data = train_df
        self.val_data = val_df
        
        return True
        
    def get_processed_data(self):
        """Return the processed train and validation datasets"""
        return self.train_data, self.val_data
        
    def format_triplet(self, row):
        """Format a single triplet into the required input format"""
        return {
            'input': f"subject: {row['subject']} predicate: {row['predicate']} object: {row['object']}",
            'target': row['text']
        } 