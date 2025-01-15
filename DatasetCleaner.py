# This cleaner is from a previous project. 
# For more info, please read the README file.
# Also, you can check this link for the original: https://github.com/alexB04676/SentimentAnalysis

import pandas as pd
import re
from tqdm import tqdm

class TextPreprocessor:
    def __init__(self):
        """Initialize the preprocessor with options to remove stopwords and lemmatize."""
        pass
    
    def clean_text(self, text):
        """Convert text to lowercase and remove special characters to clean the text for tokenization."""
        if not isinstance(text, str):
            return text
        text = text.lower()  # Convert text to lowercase to standardize the format for processing
        text = re.sub(r"\W", " ", text)  # Remove special characters
        return text

    def preprocess_dataframe(self, df, text_column="text"):
        """Apply preprocessing to a Pandas DataFrame."""
        tqdm.pandas()  # Enable progress bars for Pandas operations
        df[text_column] = df[text_column].progress_apply(self.preprocess_text)
        return df

    def save_dataframe(self, df, output_path):
        """Write the cleaned dataset to a new JSONL file for future use."""
        df.to_csv(output_path, orient='records', lines=True)
        print(f"Data saved to {output_path}")