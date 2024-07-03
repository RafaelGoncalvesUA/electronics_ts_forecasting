import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor, REGRESSORS

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape, Dropout, LeakyReLU, Conv1D, Lambda
from keras.callbacks import EarlyStopping

import argparse
import time
from utils.vae import CustomVAE
from utils.pca import CustomPCA

import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_filepath", type=str, required=True, help="Input data file (.csv)")
parser.add_argument("-t", "--test_filepath", type=str, required=True, help="Test data file (.csv)")
parser.add_argument("-g", "--global_model", action="store_true", help="Global model (if True, the data_filepath is the folder containing the data splits)")
parser.add_argument("-f", "--fine_tune", type=int, default=0, help="Fine-tune the model (0 = No, 1 = Yes, -1 = Only test)")
parser.add_argument("-o",
    "--output",
    type=str,
    required=True,
    help="File where the output will be appended (.csv)",
)
parser.add_argument("-m", "--model", type=str, required=True, help="Model")
parser.add_argument("-l", "--lookback", type=int, default=1, help="Lookback window")
parser.add_argument("-w", "--forecast_window", type=int, default=1, help="Forecast window")
parser.add_argument(
    "-r",
    "--dimensionality_reduction",
    type=str,
    default=None,
    help="Dimensionality reduction method",
)
args = parser.parse_args()

selected_model = args.model
LOOKBACK = args.lookback
OUT_STEPS = args.forecast_window
DIMENSIONALITY_REDUCTION = args.dimensionality_reduction
IS_GLOBAL_MODEL = args.global_model
FINE_TUNE = args.fine_tune

VARIABLES_TO_FORECAST = [
    "Equipment Electric Power (kWh)", # "non_shiftable_load",
    "DHW Heating (kWh)", # "dhw_demand",
    "Cooling Load (kWh)", # "cooling_demand",
    "Solar Generation (W/kW)", # "solar_generation",
    "Carbon Intensity (kg_CO2/kWh)" # "carbon_intensity",
]


def feature_engineering(df, to_forecast):
    columns = df.columns

    # create lag features
    for i in range(1, LOOKBACK):
        for f in columns:
            df[f"{f}_lag{i}"] = df[f].shift(i).values

    # create yhat for the next 48 hours
    for i in range(1, OUT_STEPS + 1):
        df[f"{to_forecast}_future{i}"] = df[to_forecast].shift(-i).values

    # drop rows with NaN values
    df.dropna(inplace=True)

    df.index = pd.RangeIndex(len(df.index))

    return df, columns


def train_preprocessing(df_arr, to_forecast):
    # create a dataframe to then concatenate all the dataframes
    res = pd.DataFrame()

    for df in df_arr:
        df, columns = feature_engineering(df, to_forecast)
        res = pd.concat([res, df])

    scaler = StandardScaler()
    res = scaler.fit_transform(res)

    print(f"Feature engineering done for {to_forecast}, shape: {res.shape}")

    lag_features = [f"{f}_lag{i}" for i in range(1, LOOKBACK) for f in columns]
    future_features = [f"{to_forecast}_future{i}" for i in range(1, OUT_STEPS + 1)]

    all_columns = columns.tolist() + lag_features + future_features

    return pd.DataFrame(res, columns=all_columns), scaler
    

def test_preprocessing(df, to_forecast, scaler=None):
    df, columns = feature_engineering(df, to_forecast)

    # reuse scaler if provided
    if scaler is None:
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
    else:
        df = scaler.transform(df)

    print(f"Feature engineering done for {to_forecast}, shape: {df.shape}")

    lag_features = [f"{f}_lag{i}" for i in range(1, LOOKBACK) for f in columns]
    future_features = [f"{to_forecast}_future{i}" for i in range(1, OUT_STEPS + 1)]

    all_columns = columns.tolist() + lag_features + future_features

    return pd.DataFrame(df, columns=all_columns), scaler


def ann_model(model_filename):
    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
            Dropout(0.35),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(OUT_STEPS, activation="linear"),
        ]
    )

    # load model if exists to fine-tune
    if FINE_TUNE != 0: # load model
        print(f"Loading {model_filename}...")
        model.load_weights(model_filename)

    model.compile(loss="mse", optimizer="adam", metrics=["mae"])

    if FINE_TUNE != -1: # fit model
        print("Training model...")
        callbacks = [
            EarlyStopping(
                monitor="loss",
                min_delta=0.0001,
                patience=30,
                mode="auto",
                restore_best_weights=True,
            )
        ]

        model.fit(
            X_train,
            Y_train,
            epochs=200,
            verbose=1,
            callbacks=callbacks,
        )

    mse = model.evaluate(X_test, Y_test, verbose=0)[0]
    print(f"MSE: {mse}")

    model.save(model_filename)
    print(f"Model saved to {model_filename}")

    return mse, "ANN"


def lazy_regressor(model_filename):
    if FINE_TUNE == -1: # does not support actual fine-tuning (only test)
        print(f"Loading model from {model_filename.replace('.keras', '.pkl')}...")
        with open(model_filename.replace(".keras", ".pkl"), "rb") as f:
            best_model = pickle.load(f)
            scores = {"mean_absolute_error": mean_absolute_error(Y_test, best_model.predict(X_test))}
            scores_df = pd.DataFrame(scores, index=[selected_model])

            return scores_df.iloc[0].mean_absolute_error, selected_model

    model = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=mean_absolute_error)
    scores, _ = model.fit(X_train, X_test, Y_train, Y_test)
    scores_df = pd.DataFrame(scores)

    regressors_lst = dict(REGRESSORS)
    best_model_name = scores_df.index[0]
    best_model = regressors_lst[best_model_name]()

    best_model.fit(X_train, Y_train)

    with open(model_filename.replace(".keras", ".pkl"), "wb") as f:
        pickle.dump(best_model, f)

    print(f"Model saved to {model_filename.replace('.keras', '.pkl')}")

    return scores_df.iloc[0].mean_absolute_error, best_model_name


available_models = {
    "ann": ann_model,
    "lazy": lazy_regressor,
}

input_filename = args.data_filepath.split("/")[-1].split(".")[0]

if IS_GLOBAL_MODEL:
    all_global_files = [f for f in os.listdir(args.data_filepath) if "global.csv" in f]
    raw_train_data = [pd.read_csv(f"{args.data_filepath}/{f}") for f in all_global_files]
    raw_test_data = pd.read_csv(args.test_filepath)
else:
    raw_train_data = [pd.read_csv(args.data_filepath)]
    raw_test_data = pd.read_csv(args.test_filepath)

results = []

for f in VARIABLES_TO_FORECAST:
    model_filename = f"models/{selected_model}_{f}_l{LOOKBACK}_f{OUT_STEPS}.keras"

    train_data = [df.copy() for df in raw_train_data]
    test_data = raw_test_data.copy()

    train_data, scaler = train_preprocessing(train_data, f)
    test_data, _ = test_preprocessing(test_data, f, scaler)

    if DIMENSIONALITY_REDUCTION == "vae":
        vae = CustomVAE(base_dir="models")

        train_data = vae.fit(
            train_data,
            to_forecast=f,
            lookback=LOOKBACK,
            out_steps=OUT_STEPS,
            latent_dim=5,
            train_again=not os.path.exists(f'models/encoder_{f.replace("/", "")}_l{LOOKBACK}_f{OUT_STEPS}.keras')
        )

        test_data = vae.predict(test_data)

    elif DIMENSIONALITY_REDUCTION == "pca":
        pca = CustomPCA()
        train_data = pca.fit(
            train_data,
            to_forecast=f,
            out_steps=OUT_STEPS
        )

        test_data = pca.predict(test_data)

    X_train = train_data.drop([f"{f}_future{i}" for i in range(1, OUT_STEPS + 1)], axis=1)
    Y_train = train_data[[f"{f}_future{i}" for i in range(1, OUT_STEPS + 1)]]
    X_test = test_data.drop([f"{f}_future{i}" for i in range(1, OUT_STEPS + 1)], axis=1)
    Y_test = test_data[[f"{f}_future{i}" for i in range(1, OUT_STEPS + 1)]]

    mse, model_name = available_models[selected_model](model_filename)

    results.append(
        {
            "var": f,
            "input": input_filename,
            "model": model_name,
            "lookback": LOOKBACK,
            "forecast": OUT_STEPS,
            "mse": mse,
            "timestamp": time.time(),
        }
    )

results = pd.DataFrame(results)
results.to_csv(args.output, mode="a", header=False, index=False)
