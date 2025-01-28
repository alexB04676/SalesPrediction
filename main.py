import pandas as pd
from DatasetCleaner import Preprocessor
import os

# read the dataset with relative paths and error handling
try:
    data_path = os.path.join(os.getcwd(), "beauty_cosmetics_products_sales.csv")
    df = pd.read_csv(data_path, header = 0)
    if df.empty:
        raise ValueError("Dataset is empty. ")
except FileNotFoundError:
    print("Error: cleaned_dataset.jsonl not found. ")
    exit()
except ValueError as e:
    print(e)
    exit()

# initiate preprocessor
preprocess = Preprocessor()

"""
# block of code to see the min/max of our numerical columns and decide on the approach for scaling
# list of our numerical columns
columns = ['Price_USD', 'Number_of_Reviews', 'Rating', 'Product_Size']

# iterate through the list to find and print the min/max of each column to decide on the approach for scaling
for col in columns:
    print(f"{col}:")
    print(df[col].agg(['min', 'max']))
    print()
"""

# drop irrelavent columns
df = preprocess.drop(df, columns= ["Product_Name", "Packaging_Type"])

# use normalization scaling method on numerical columns 
df = preprocess.normalize(df, columns= ['Number_of_Reviews', 'Price_USD'])

# get a list of non-numerical columns' values to decide on the approach for encoding
"""list = preprocess.unique_items_list(df, columns = ["Product_Name", "Brand", "Category", "Usage_Frequency", "Product_Size", "Skin_Type", 
"Gender_Target", "Packaging_Type", "Main_Ingredient", "Cruelty_Free", "Country_of_Origin"])"""

df = preprocess.OneHotEncoder(df, columns=["Usage_Frequency", "Product_Size", "Skin_Type", "Gender_Target", "Main_Ingredient", "Country_of_Origin"])

print(df.shape)