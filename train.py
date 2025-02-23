"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data import process_data
from model import model
from keras import Model
from keras.api.callbacks import EarlyStopping
from keras.api.losses import MeanSquaredError
warnings.filterwarnings("ignore")

def train_model(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """
    print(f"Compiling {name} model...")
    model.compile(loss=MeanSquaredError(), optimizer="rmsprop", metrics=['mape'])

    print(f"Starting training for {name}...")
    print(f"Input shape: {X_train.shape}")
    print(f"Output shape: {y_train.shape}")

    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        verbose=1)  # Added verbose=1 to see training progress

    print(f"Saving {name} model...")
    model.save('model/' + name + '.keras')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + name + ' loss.csv', encoding='utf-8', index=False)
    print(f"Training completed for {name}")

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
    temp = X_train

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
        m.fit(temp, y_train,
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

    print("Training final SAES model...")
    train_model(saes, X_train, y_train, name, config)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train.")
    args = parser.parse_args()

    print(f"Starting training process for {args.model} model...")

    lag = 12
    config = {"batch": 256, "epochs": 2}
    file1 = 'Scats2006.xls'
    file2 = 'Scats2006.xls'

    print("Loading and processing data...")
    try:
        X_train, y_train, _, _, _ = process_data(file1, file2, lag)
        print("Data loaded successfully")
        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    if args.model == 'lstm':
        print("Preparing LSTM model...")
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_lstm([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    elif args.model == 'gru':
        print("Preparing GRU model...")
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_gru([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    elif args.model == 'saes':
        print("Preparing SAES model...")
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        m = model.get_saes([12, 400, 400, 400, 1])
        train_saes(m, X_train, y_train, args.model, config)
    else:
        print(f"Unknown model type: {args.model}")

if __name__ == '__main__':
    main(sys.argv)