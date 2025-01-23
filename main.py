import pandas as pd
from DatasetCleaner import Preprocessor

# open our dataset
df = pd.read_csv("C:/Users/ali/Projects/SalesPrediction/beauty_cosmetics_products_sales.csv")

# initiate preprocessor
preprocess = Preprocessor()

# list of our numerical columns
columns = ['Price_USD', 'Number_of_Reviews', 'Rating', 'Product_Size']

# iterate through the list to find and print the min/max of each column to decide on the approach for scaling
for col in columns:
    print(f"{col}:")
    print(df[col].agg(['min', 'max']))
    print()

# drop irrelavent columns
df = preprocess.drop(df, columns= ["Product_Name", "Packaging_Type"])

print(df.columns)
print(df.shape)