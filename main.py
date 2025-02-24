import pandas as pd
from DatasetCleaner import Preprocessor
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# read the dataset with relative paths and error handling
try:
    data_path = os.path.join(os.getcwd(), "Sampled_dataset.csv")
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
df = preprocess.save_dataframe(df, "C:/Users/ali/Projects/SalesPrediction/Sampled_dataset.csv")

df = preprocess.columns_drop(df, columns = ["id", "url", "region_url", "VIN", "image_url", "description", "county", "lat", "long", "posting_date"])

minmax = preprocess.min_max_finder(df, columns = ["price", "odometer", "year"])

df = df.dropna()
df = preprocess.value_rows_remover(df, 0, columns=["price"])
df = preprocess.value_remover(df, value= [(500,200000), 500000, 1950], columns=["price", "odometer", "year"], mode = ["range", "below", "above"])

df = preprocess.columns_drop(df, columns= "region")

df = preprocess.normalize(df, columns= ["price", "odometer", "year"])
items = preprocess.unique_items_list(df, columns= "model")
df = preprocess.FrequencyEncoding(df, columns= "model")
df = preprocess.save_dataframe(df, "C:/Users/ali/Projects/SalesPrediction/Sampled_dataset.csv", file_format= "csv")"""

"""df = preprocess.OneHotEncoder(df, columns= ["condition", "cylinders", "fuel", "title_status", "transmission", "drive", "size", "type", "paint_color"])
df = preprocess.FrequencyEncoding(df, columns=["manufacturer"])
df = preprocess.TargetEncoder(df, columns=["state"], target= "price")
df = preprocess.columns_drop(df, columns=["manufacturer", "model"])
df = preprocess.save_dataframe(df, output_path= "C:/Users/ali/Projects/SalesPrediction/Sampled_dataset.csv", file_format= "csv")"""

X = df.drop("price", axis= 1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

dtrain_reg = xgb.DMatrix(X_train, y_train)
dtest_reg = xgb.DMatrix(X_test, y_test)

params = {"booster": "gbtree", "objective": "reg:squarederror", "device": "gpu", "eval_metric": "rmse", "subsample": 0.6}

n = 100
model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
)

preds = model.predict(dtest_reg)

scaler = MinMaxScaler()
rmse = root_mean_squared_error(y_test, preds)

print(f"RMSE of the base model: {rmse:.3f}")