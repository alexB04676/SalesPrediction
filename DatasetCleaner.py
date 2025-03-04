# This cleaner is from a previous project. 
# For more info, please read the README file.
# Also, you can check this link for the original: https://github.com/alexB04676/SentimentAnalysis

import pandas as pd
import re
from tqdm import tqdm
from typing import Union
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder


class Preprocessor:
    
    tqdm.pandas()
    
    def __init__(self):
        pass
    
    def clean_text(self, text):
        """Convert text to lowercase and remove special characters to clean the text for tokenization."""
        if not isinstance(text, str):
            return text
        text = text.lower()  # Convert text to lowercase to standardize the format for processing
        text = re.sub(r"\W", " ", text)  # Remove special characters
        return text

    def rows_sampling(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if not isinstance(n, int):
            raise TypeError("The number of rows should be an integer")
        
        if isinstance(df, str):
            df = pd.read_csv(df, index_col= False)
        
        return df.sample(n=n, random_state=42, ignore_index= True)
       
    def preprocess_dataframe(self, df: pd.DataFrame, text_column="text") -> pd.DataFrame:
        """Apply preprocessing to a Pandas DataFrame."""
        tqdm.pandas()  # Enable progress bars for Pandas operations
        df[text_column] = df[text_column].progress_apply(self.preprocess_text)
        return df

    def value_rows_remover(self, df: pd.DataFrame, value: int, columns: Union[str, list]):
        
        if isinstance(columns, str):
            columns = [columns]
        
        if not isinstance(value, int):
            raise TypeError("The value should either be an integer.")
            
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Column(s) {missing_columns} not found in DataFrame.")

        # Replace the given value with NaN only in the specified columns
        df[columns] = df[columns].progress_apply(lambda x: x.replace(value, pd.NA))

        # Drop only NaN values in the specified columns, but keep the other data
        df = df.dropna(subset=columns)

        return df

        
    def columns_drop(self, df: pd.DataFrame, columns: Union[list, str]) -> pd.DataFrame:
        
        if isinstance(columns, str):
            columns = [columns]
        
        # Check if columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Column(s) {missing_columns} not found in DataFrame.")
        
        # Drop specified columns
        df = df.drop(columns=columns, axis=1)
        return df
    
    def value_remover(self, df: pd.DataFrame, value: Union[int, list], columns: Union[str, list], mode: Union[str, list]) -> pd.DataFrame:
        
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(value, int) or isinstance(value, tuple):
            value = [value]
        if isinstance(mode, str):
            mode = [mode]

        # Check if all lists have the same length
        if not (len(columns) == len(value) == len(mode)):
            raise ValueError("Columns, values, and modes must have the same length.")

        
        for col, val, mod in zip(columns, value, mode):
            
            if isinstance(val, tuple) and len(val) == 2:
                mod == "range"
                df = df[(df[col] >= val[0]) & (df[col] <= val[1])]
            elif isinstance(val, int):
                if mod == "below":
                    df = df[df[col] <= val]
                elif mod == "above":
                    df = df[df[col] >= val]
                else:
                    raise ValueError("""Please type "above", "below" or "range" for the mode to start removing""")

            else:
                raise ValueError(f"Invalid value type for column '{col}'. Must be an int or tuple.")
            
        return df

    def normalize(self, df: pd.DataFrame, columns: Union[list, str]) -> pd.DataFrame:
        
        if isinstance(columns, str):
            columns = [columns]
        
        # Check if columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Column(s) {missing_columns} not found in DataFrame.")
        
        
        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df
    
    def unique_items_list(self, df: pd.DataFrame, columns: Union[list, str], count: bool):
        
        result = {}  # Dictionary to store unique items for each column

        if isinstance(columns, str):
            columns = [columns]
        
        # Check if columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Column(s) {missing_columns} not found in DataFrame.")
        
        for col in columns:
            if col in df.columns:  # Ensure the column exists in the DataFrame
                unique_items = df[col].dropna().astype(str).str.strip().unique()
                result[col] = sorted(unique_items)  # Sort the unique items alphabetically
        
        if count == True:
            for col in result:
                print(f"{col}: {len(result[col])}")
        else:
            pass
        
        print(result)
        return result

    def min_max_finder(self, df: pd.DataFrame, columns: Union[list, str]) -> list:
        
        if isinstance(columns, str):
            columns = [columns]
            
       # Check if columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Column(s) {missing_columns} not found in DataFrame.")
        
        for col in columns:
            print(f"{col}:")
            print(df[col].agg(['min', 'max']))
            print()

    def OneHotEncoder(self, df: pd.DataFrame, columns: Union[list, str]) -> pd.DataFrame:
        
        if isinstance(columns, str):
            columns = [columns]
        
        # Check if columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Column(s) {missing_columns} not found in DataFrame.")
        
        # Initialize OneHotEncoder (ignores unknown categories and outputs a pandas DataFrame)
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(transform="pandas")
        
        # Transform the specified columns
        ohe_transformed = ohe.fit_transform(df[columns])
        
        # Merge the new one-hot encoded columns with the original DataFrame
        df = pd.concat([df.drop(columns, axis=1), ohe_transformed], axis=1)
        return df
    
    def TargetEncoder(self, df: pd.DataFrame, columns: Union[list, str], target = str) -> pd.DataFrame :
        
        if isinstance(columns, str):
            columns = [columns]
        
        if isinstance(target, str):
            target = [target]
        
        # Check if columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Column(s) {missing_columns} not found in DataFrame.")
        
        te = TargetEncoder(smoothing= 4.0, handle_unknown= "value", min_samples_leaf= 10.0, handle_missing= "value")
        
        te_transformed = te.fit_transform(df[columns], df[target])
        
        # Merge the new one-hot encoded columns with the original DataFrame
        df = pd.concat([df.drop(columns, axis=1), te_transformed], axis=1)
        return df
    
    def FrequencyEncoding(self, df: pd.DataFrame, columns: Union[list, str]):
        
        if isinstance(columns, str):
            columns = [columns]
        
        # Check if columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns]
        
        if missing_columns:
            raise KeyError(f"Column(s) {missing_columns} not found in DataFrame.")

        for col in columns:
            df[f"{col}_freq"] = df[col].map(df[col].value_counts(normalize=True))
        
        return df
        
    def save_dataframe(self, df, output_path: str, file_format: str):
        """Write the cleaned dataset to a new file for future use."""
        
        if file_format.lower() == "csv":
            df.to_csv(output_path, index = False)
            print(f"Data saved to {output_path}")
            
        elif file_format.lower() == "json":
            df.to_json(output_path, index = False)
            print(f"Data saved to {output_path}")
            
        elif file_format.lower() == "jsonl":
            df.to_json(output_path, index = False, lines = True)
            print(f"Data saved to {output_path}")
            
        elif file_format.lower() == "excel":
            df.to_excel(output_path, index = False)
            print(f"Data saved to {output_path}")
        
        else:
            print("please enter a viable file format (CSV, JSON, JSONL, Excel)")