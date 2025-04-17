from DatasetCleaner import MappingManager
from DatasetCleaner import Preprocessor
from cleaning_process import cleaning
from model_training import training
import argparse
import pandas as pd
import joblib
import xgboost as xgb
import os
import numpy as np

# clean = cleaning()

# train = training()

preprocess = Preprocessor()
mapper = MappingManager()

def get_user_input():
    
    parser = argparse.ArgumentParser(description="Car Price Prediction CLI")
    parser.add_argument("--manufacturer", type=str, help="Manufacturer")
    parser.add_argument("--model", type=str, help="Model")
    parser.add_argument("--year", type=int, help="Model Year")
    parser.add_argument("--mileage", type=int, help="Mileage (By Miles)")
    args = parser.parse_args()

    # Ask for missing inputs interactively
    if args.manufacturer is None:
        args.manufacturer = input("Enter Manufacturer: ").strip()
    if args.model is None:
        args.model = input("Enter Model: ").strip()
    if args.year is None:
        args.year = int(input("Enter Model Year: ").strip())
    if args.mileage is None:
        args.mileage = int(input("Enter Mileage (By Miles): ").strip())
    
    return vars(args)
    
def process_input(df: pd.DataFrame):
    
    try:
        data_path = os.path.join(os.getcwd(), "Sampled_dataset.csv")
        df = pd.read_csv(data_path, header = 0)
        if df.empty:
            raise ValueError("Dataset is empty. ")
    except FileNotFoundError:
        print("Error: file not found. Make sure you've went through the preprocessing step? ")
        exit()
    except ValueError as e:
        print(e)
        exit()
        
    input_data = pd.DataFrame([np.zeros(len(df.columns))], columns=df.columns)
    
    if "price" in input_data.columns:
        input_data = input_data.drop(columns=["price"])

    year_scaler = joblib.load("scalers/year_scaler.pkl")
    mileage_scaler = joblib.load("scalers/odometer_scaler.pkl")
    manufacturer_mapping= mapper.load_mapping("manufacturer")
    model_mapping = mapper.load_mapping("model")
    
    manufacturer_converted = manufacturer_mapping.get(input_data["manufacturer_freq"].iloc[0], 0.0001)
    model_converted = model_mapping.get(input_data["model_freq"].iloc[0], 0.0001)
    year_converted = year_scaler.transform(pd.DataFrame([[user_input["year"]]], columns=["year"]))
    mileage_converted = mileage_scaler.transform(pd.DataFrame([[user_input["mileage"]]], columns=["odometer"]))
    
    input_data.at[0, "manufacturer_freq"] = manufacturer_converted
    input_data.at[0, "model_freq"] = model_converted
    input_data.at[0, "year"] = year_converted[0][0]
    input_data.at[0, "odometer"] = mileage_converted[0][0]
    
    return input_data

def predict_price(model: joblib, df: pd.DataFrame):
    
    dmatrix = xgb.DMatrix(df)  # Convert DataFrame to DMatrix
    predicted_price = model.predict(dmatrix)[0]  # Predict price
    error_margin = 0.022 * predicted_price  # Compute ±2.2% range
    return predicted_price, predicted_price - error_margin, predicted_price + error_margin


if __name__ == "__main__":
    # Load trained model
    model = joblib.load("XGBoostSP.joblib")

    # Get user input
    user_input = get_user_input()

    # Preprocess input using mappings
    df_input = process_input(user_input)

    # Make prediction
    predicted_price, lower, upper = predict_price(model, df_input)

    print(f"Predicted Price: ${predicted_price:,.2f}")
    print(f"Estimated Range: ${lower:,.2f} - ${upper:,.2f}")