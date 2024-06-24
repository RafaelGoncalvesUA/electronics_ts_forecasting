import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape, Dropout, LeakyReLU, Conv1D, Lambda
from keras.callbacks import EarlyStopping

import argparse
import os
import time
from utils.vae import CustomVAE
from utils.pca import CustomPCA

# timestamp append
# program -m model -d data.csv -o output.csv

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_filepath", type=str, required=True, help="Input data file (.csv)")
parser.add_argument("-o",
    "--output",
    type=str,
    required=True,
    help="File where the output will be appended (.csv)",
)
parser.add_argument("-m", "--model", type=str, required=True, help="Model")
parser.add_argument("-l", "--lookback", type=int, default=1, help="Lookback window")
parser.add_argument("-f", "--forecast", type=int, default=1, help="Forecast window")
parser.add_argument(
    "-r",
    "--dimensionality_reduction",
    type=str,
    default=None,
    help="Dimensionality reduction method",
)
args = parser.parse_args()

data = pd.read_csv(args.data_filepath)
selected_model = args.model
LOOKBACK = args.lookback
OUT_STEPS = args.forecast
DIMENSIONALITY_REDUCTION = args.dimensionality_reduction

VARIABLES_TO_FORECAST = [
    "non_shiftable_load",
    "dhw_demand",
    "cooling_demand",
    "solar_generation",
    "carbon_intensity",
]


def preprocessing(df, to_forecast, scaler=None):
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

    # reuse scaler if provided
    if scaler is None:
        scaler = StandardScaler()
        data = scaler.fit_transform(df)
    else:
        data = scaler.transform(df)

    print(f"Preprocessing done for {to_forecast}, shape: {data.shape}")

    return df, scaler


def ann_model(X_train, Y_train):
    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
            Dropout(0.35),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(OUT_STEPS, activation="linear"),
        ]
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.00001,
            patience=10,
            mode="auto",
            restore_best_weights=True,
        )
    ]

    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    model.fit(
        X_train,
        Y_train,
        epochs=200,
        verbose=1,
        callbacks=callbacks,
    )

    return model


available_models = {
    "ann": ann_model,
}

input_filename = args.data_filepath.split("/")[-1].split(".")[0]

train_data, test_data = train_test_split(data, test_size=0.3, shuffle=False)
results = []

for f in VARIABLES_TO_FORECAST:
    train_data, scaler = preprocessing(train_data, f)
    test_data, _ = preprocessing(test_data, f, scaler)

    if DIMENSIONALITY_REDUCTION == "vae":
        vae = CustomVAE(base_dir="models")
        data = vae.apply(
            data,
            f,
            latent_dim=5,
            train_again=not os.path.exists(f"models/encoder_{f}.keras"),
        )

    elif DIMENSIONALITY_REDUCTION == "pca":
        pca = CustomPCA()
        data = pca.apply(data, f)

    X_train = X_train.drop([f"{f}_future{i}" for i in range(1, OUT_STEPS + 1)], axis=1)

    Y_train = train_data[[f"{f}_future{i}" for i in range(1, OUT_STEPS + 1)]]

    model = available_models[selected_model](X_train, Y_train)

    X_test = test_data.drop([f"{f}_future{i}" for i in range(1, OUT_STEPS + 1)], axis=1)

    Y_test = test_data[[f"{f}_future{i}" for i in range(1, OUT_STEPS + 1)]]

    mse = model.evaluate(X_test, Y_test, verbose=0)[0]

    results.append(
        {
            "var": f,
            "input": input_filename,
            "model": selected_model,
            "lookback": LOOKBACK,
            "forecast": OUT_STEPS,
            "mse": mse,
            "timestamp": time.time(),
        }
    )

    continue

