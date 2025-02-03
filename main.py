import pandas as pd
from DatasetCleaner import Preprocessor
import os

# read the dataset with relative paths and error handling
try:
    data_path = os.path.join(os.getcwd(), "vehicles.csv")
    df = pd.read_csv(data_path, header = 0)
    if df.empty:
        raise ValueError("Dataset is empty. ")
except FileNotFoundError:
    print("Error: file not found. Make sure you're typing the right name?")
    exit()
except ValueError as e:
    print(e)
    exit()

# initiate preprocessor
preprocess = Preprocessor()

"""df = preprocess.rows_sampling(df, 420000)
df = preprocess.save_dataframe(df, "C:/Users/ali/Projects/SalesPrediction/Sampled_dataset.csv")"""

df = pd.read_csv("C:/Users/ali/Projects/SalesPrediction/Sampled_dataset.csv")
df = preprocess.columns_drop(df, columns = ["id", "url", "region_url", "VIN", "image_url", "description", "county", "lat", "long", "posting_date"])
df = df.dropna()
df = preprocess.value_rows_remover(df, 0, columns=["price"])
df = preprocess.save_dataframe(df, "C:/Users/ali/Projects/SalesPrediction/Sampled_dataset2.csv")