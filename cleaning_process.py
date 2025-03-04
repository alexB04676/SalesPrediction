from DatasetCleaner import Preprocessor
import os
import pandas as pd

def cleaning():
    
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

    preprocess = Preprocessor()

    df = preprocess.rows_sampling(df, 420000)
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
    df = preprocess.save_dataframe(df, "C:/Users/ali/Projects/SalesPrediction/Sampled_dataset.csv", file_format= "csv")

    df = preprocess.OneHotEncoder(df, columns= ["condition", "cylinders", "fuel", "title_status", "transmission", "drive", "size", "type", "paint_color"])
    df = preprocess.FrequencyEncoding(df, columns=["manufacturer"])
    df = preprocess.TargetEncoder(df, columns=["state"], target= "price")
    df = preprocess.columns_drop(df, columns=["manufacturer", "model"])
    df = preprocess.save_dataframe(df, output_path= "C:/Users/ali/Projects/SalesPrediction/Sampled_dataset.csv", file_format= "csv")