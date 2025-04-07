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
import os
import json
import joblib

class MappingManager:
    
    def __init__(self, save_path="mappings"):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def save_mapping(self, column_name, mapping, format: str = "joblib"):
        format = format.lower().strip()
        file_path = os.path.join(self.save_path, f"{column_name}_mapping.{format}")

        if format == "json":
            with open(file_path, "w") as f:
                json.dump(mapping, f)
            print(f"✅ Mapping saved: {file_path}")

        elif format == "joblib":
            file_path = os.path.join(self.save_path, f"{column_name}_mapping.pkl")
            joblib.dump(mapping, file_path)
            print(f"✅ Mapping saved: {file_path}")

        else:
            print("❌ Please enter a valid format ('joblib' or 'json').")

    def load_mapping(self, column_name, format: str = "joblib"):
        format = format.lower().strip()
        file_path = os.path.join(self.save_path, f"{column_name}_mapping.{format}")

        if format == "json":
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    return json.load(f)
            else:
                print(f"❌ Mapping '{column_name}' (JSON) not found in '{self.save_path}'.")

        elif format == "joblib":
            file_path = os.path.join(self.save_path, f"{column_name}_mapping.pkl")
            if os.path.exists(file_path):
                return joblib.load(file_path)
            else:
                print(f"❌ Mapping '{column_name}' (Joblib) not found in '{self.save_path}'.")

        else:
            print("❌ Please enter a valid format ('joblib' or 'json').")

    def apply_mapping(self, series, mapping):
        return series.map(mapping)


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

    def normalize(self, df: pd.DataFrame, columns: Union[list, str], mapping_return: bool = False, mapping_format: str = "joblib") -> pd.DataFrame:
        # Convert a single column name to a list for consistent processing
        if isinstance(columns, str):
            columns = [columns]

        # Check if all specified columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Column(s) {missing_columns} not found in DataFrame.")

        # Initialize the scaler and mapping manager
        scaler = MinMaxScaler()
        mapping_manager = MappingManager() if mapping_return else None
        
        # Loop through each column to apply normalization
        for col in columns:
            # Perform Min-Max Scaling and reshape to fit the scaler's expected input
            normalized = scaler.fit_transform(df[[col]])

            # Save the scaler object as mapping if requested
            if mapping_return:
                if mapping_format.lower().strip() == "json":
                    mapping_manager.save_mapping(col, scaler, format="json")
                
                else:
                    mapping_manager.save_mapping(col, scaler, format="joblib")

            # Update the DataFrame column with the normalized values
            df[col] = normalized

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

    def OneHotEncoder(self, df: pd.DataFrame, columns: Union[list, str], mapping_return: bool = False) -> pd.DataFrame:
        
        # Convert a single column name to a list for consistent processing
        if isinstance(columns, str):
            columns = [columns]

        # Check if all specified columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Column(s) {missing_columns} not found in DataFrame.")

        # Initialize the OneHotEncoder with settings to handle unknown and missing values
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(transform="pandas")
        
        # Initialize the mapping manager
        mapping_manager = MappingManager() if mapping_return else None
        
        # Loop through each column to apply one-hot encoding
        for col in columns:
            # Transform the data using the encoder and store as a DataFrame
            ohe_transformed = ohe.fit_transform(df[[col]])

            # Save the fitted encoder object as mapping if requested
            if mapping_return:
                mapping_manager.save_mapping(col, ohe)

            # Concatenate the one-hot encoded columns with the original DataFrame
            df = pd.concat([df.drop(col, axis=1), ohe_transformed], axis=1)

        return df


    def TargetEncoder(self, df: pd.DataFrame, columns: Union[list, str], target: str, mapping_return: bool = False) -> pd.DataFrame:
        # Convert a single column name to a list for consistent processing
        if isinstance(columns, str):
            columns = [columns]

        # Check if all specified columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Column(s) {missing_columns} not found in DataFrame.")

        # Initialize the TargetEncoder with configurations
        te = TargetEncoder(smoothing=4.0, handle_unknown="value", min_samples_leaf=10.0, handle_missing="value")

        # Initialize the mapping manager
        mapping_manager = MappingManager() if mapping_return else None
        
        # Apply target encoding using the specified columns and target variable
        te_transformed = te.fit_transform(df[columns], df[target])

        # Save the fitted encoder object as mapping if requested
        if mapping_return:
            mapping_manager.save_mapping(",".join(columns), te)

        # Merge the target encoded columns back into the original DataFrame
        df = pd.concat([df.drop(columns, axis=1), te_transformed], axis=1)

        return df

    
    def FrequencyEncoder(self, df: pd.DataFrame, columns: Union[list, str], mapping_return: bool = False) -> pd.DataFrame:
        
        # Convert a single column name to a list for consistent processing
        if isinstance(columns, str):
            columns = [columns]

        # Check if all specified columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Column(s) {missing_columns} not found in DataFrame.")

        # Initialize the mapping manager
        mapping_manager = MappingManager() if mapping_return else None
        
        for col in columns:
            # Calculate frequency of each unique value
            df[f"{col}_freq"] = df[col].map(df[col].value_counts(normalize=True))
            freq = df[col].value_counts(normalize=True).to_dict()

            # Save the frequency mapping if requested
            if mapping_return:
                mapping_manager.save_mapping(col, freq)

            # Map each value to its calculated frequency
            df[col] = df[col].map(freq)

        return df


    def save_dataframe(self, df, output_path: str, file_format: str):
        """Write the cleaned dataset to a new file for future use."""
        
        if file_format.lower().strip() == "csv":
            df.to_csv(output_path, index = False)
            print(f"Data saved to {output_path}")
            
        elif file_format.lower().strip() == "json":
            df.to_json(output_path, index = False)
            print(f"Data saved to {output_path}")
            
        elif file_format.lower().strip() == "jsonl":
            df.to_json(output_path, index = False, lines = True)
            print(f"Data saved to {output_path}")
            
        elif file_format.lower().strip() == "excel":
            df.to_excel(output_path, index = False)
            print(f"Data saved to {output_path}")
        
        else:
            print("please enter a viable file format (CSV, JSON, JSONL, Excel)")
            
        return df