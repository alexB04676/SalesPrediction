import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from joblib import dump

def training():
    
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

    X = df.drop("price", axis= 1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    dtrain_reg = xgb.DMatrix(X_train, y_train)
    dtest_reg = xgb.DMatrix(X_test, y_test)

    params = {"booster": "gbtree", "objective": "reg:squarederror", "device": "gpu", "eval_metric": "rmse", "subsample": 0.9
    , "max_depth": 9, "lambda": 1, "learning_rate": 0.05, "colsample_bytree": 0.7}

    model = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    num_boost_round=1200,
    )

    preds = model.predict(dtest_reg)

    rmse = root_mean_squared_error(y_test, preds)

    print(f"RMSE of the base model: {rmse:.3f}")

    dump(model, "XGBoostSP.joblib")
    print("Model Saved as 'XGBoostSP.joblib'")