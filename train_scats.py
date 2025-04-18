# from train import train_model, train_saes
import pandas as pd
import sys
import argparse
import numpy as np 
from model import model
from keras import Model
from keras.api.callbacks import EarlyStopping
from keras.api.losses import MeanSquaredError

from data import process_data, process_scats_data
LAG =12
EPOCH = 50
FILE = "Scats2006.xls"

def get_all_scats_number(file):
    df = pd.read_excel(file, sheet_name="Summary Of Data", header=0, skiprows=3)
    df = df[["SCATS Number"]].dropna()
    return df.astype(int).to_numpy().flatten().tolist()

def train_scats_model(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """
    print(f"Compiling {name['model']} model...")
    model.compile(loss=MeanSquaredError(), optimizer="rmsprop", metrics=['mape'])

    print(f"Starting training for {name['model']}...")
    print(f"Input shape: {X_train.shape}")
    print(f"Output shape: {y_train.shape}")

    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        verbose=1)  # Added verbose=1 to see training progress

    print(f"Saving {name['model']} model...")
    model.save(f"model/{name['model']}/{name['scat']}.keras")
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(f"model/{name['model']}/{name['scat']} loss.csv", encoding='utf-8', index=False)
    print(f"Training completed for {name['model']}")

def train_saes(models, X_train, y_train, name, config):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """
    print("Starting SAE training...")
    temp = np.copy(X_train)

    for i in range(len(models) - 1):
        print(f"Training SAE layer {i+1}")
        if i > 0:
            p = models[i - 1]
            hidden_layer = p.get_layer('hidden')
            hidden_layer_model = Model(inputs=p.input, outputs=hidden_layer.output)
            temp = hidden_layer_model.predict(temp, verbose=0)
            print(f"Generated hidden representations for layer {i+1}")

        m = models[i]
        m.compile(loss=MeanSquaredError(), optimizer="rmsprop", metrics=['mape'])

        print(f"Training autoencoder {i+1}")
        m.fit(temp, temp,
              batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05,
              verbose=1)

        models[i] = m

    saes = models[-1]
    print("Transferring weights to final model...")
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer(f'hidden{i + 1}').set_weights(weights)

    print("Fine-tuning the final SAES model...")
    saes.compile(loss=MeanSquaredError(), optimizer="adam", metrics=['mape'])
    saes.fit(X_train, y_train,
                batch_size=config["batch"],
                epochs=config["epochs"],
                validation_split=0.05,
                verbose=1)

    print("Training final SAES model...")
    train_scats_model(saes, X_train, y_train, name, config)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train.")
    args = parser.parse_args()
    config = {"batch": 256, "epochs": 470}
    print("Loading and processing data...")

    
    scats = get_all_scats_number("Scats2006.xls")
    info = {
        "model": args.model,
  
    }
    for scat in scats:
        info["scat"] = scat
        try:
            X_train, y_train, _, _, _ = process_scats_data(FILE, LAG,scat)
            print("Data loaded successfully")
            print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return
        if args.model == 'lstm':
            print("Preparing LSTM model...")
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_lstm([12, 64, 64, 1])
            train_scats_model(m, X_train, y_train, info, config)
        elif args.model == 'gru':
            print("Preparing GRU model...")
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_gru([12, 64, 64, 1])
            train_scats_model(m, X_train, y_train, info, config)
        elif args.model == 'saes':
            print("Preparing SAES model...")
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
            m = model.get_saes([12, 400, 400, 400, 1])
            train_saes(m, X_train, y_train, info, config)
        else:
            print(f"Unknown model type: {info.model}")
        
        

    

if __name__ == "__main__":
    main()