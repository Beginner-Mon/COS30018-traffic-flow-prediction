"""
Train the NN model with support for multiple SCAT sites using embeddings.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from data import process_data
from model import model
from keras.api.layers import LSTM, GRU
from keras import Model
from keras.api.layers import Input, Embedding, Flatten, Concatenate, Dense
from keras.api.callbacks import EarlyStopping
from keras.api.losses import MeanSquaredError
warnings.filterwarnings("ignore")

def train_model(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: tuple (time_series_data, site_ids), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """
    print(f"Compiling {name['model']} model...")
    model.compile(loss=MeanSquaredError(), optimizer="rmsprop", metrics=['mape'])

    print(f"Starting training for {name['model']}...")
    time_series_data, site_ids = X_train
    print(f"Time series data shape: {time_series_data.shape}")
    print(f"Site IDs shape: {site_ids.shape}")
    print(f"Output shape: {y_train.shape}")

    hist = model.fit(
        [time_series_data, site_ids], y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        verbose=1)  # Added verbose=1 to see training progress

    print(f"Saving {name['model']} model...")
    model.save(f"model/{name['model']}_multi_site.keras")
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(f"model/{name['model']}_multi_site_loss.csv", encoding='utf-8', index=False)
    print(f"Training completed for {name['model']}")

def train_saes(models, X_train, y_train, name, config):
    """train
    train the SAEs model with site embeddings.

    # Arguments
        models: Dict, contains autoencoder models and final model.
        X_train: tuple (time_series_data, site_ids), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """
    print("Starting SAE training...")
    time_series_data, site_ids = X_train
    autoencoder_models = models['autoencoders']
    final_model = models['final_model']

    temp = np.copy(time_series_data)

    # Train each autoencoder layer
    for i in range(len(autoencoder_models)):
        print(f"Training SAE layer {i+1}")
        if i > 0:
            p = autoencoder_models[i-1]
            hidden_layer = p.get_layer('hidden')
            hidden_layer_model = Model(inputs=p.input, outputs=hidden_layer.output)
            temp = hidden_layer_model.predict(temp, verbose=0)
            print(f"Generated hidden representations for layer {i+1}")

        m = autoencoder_models[i]
        m.compile(loss=MeanSquaredError(), optimizer="rmsprop", metrics=['mape'])

        print(f"Training autoencoder {i+1}")
        m.fit(temp, temp,
              batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05,
              verbose=1)

        autoencoder_models[i] = m

    # Transfer learned weights to the final model
    print("Transferring weights to final model...")
    for i in range(len(autoencoder_models)):
        weights = autoencoder_models[i].get_layer('hidden').get_weights()
        final_model.get_layer(f'hidden{i+1}').set_weights(weights)

    # Fine-tune the final model
    print("Fine-tuning the final SAES model...")
    final_model.compile(loss=MeanSquaredError(), optimizer="adam", metrics=['mape'])
    hist = final_model.fit(# <-- Capture history object
        [time_series_data, site_ids], y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        verbose=1
    )

    print("Training completed for SAES model")
    print(f"Saving {name['model']} model...")
    final_model.save(f"model/{name['model']}_multi_site.keras")
    pd.DataFrame(hist.history).to_csv(f"model/{name['model']}_loss.csv", index=False)

def process_multi_site_data(file_path, lag):
    """
    Process data for multiple SCAT sites.

    # Arguments
        file_path: String, path to data file.
        lag: Integer, length of history data to use.

    # Returns
        X_train: tuple (time_series_data, site_ids) for training.
        y_train: Target values for training.
        X_test: tuple (time_series_data, site_ids) for testing.
        y_test: Target values for testing.
        num_sites: Number of unique SCAT sites.
        site_mapping: Dictionary mapping SCAT IDs to indices.
    """
    print(f"Loading data from {file_path}...")

    # Load the data
    try:
        data = pd.read_csv(file_path)
        # Rename columns for easier handling
        data.columns = ["timestamp", "flow", "lane_points", "observed_pct", "scat_id"]
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        # Try CSV format as fallback
        try:
            data = pd.read_csv(file_path)
            data.columns = ["timestamp", "flow", "lane_points", "observed_pct", "scat_id"]
        except:
            raise Exception(f"Could not load data from {file_path}")

    print(f"Data loaded with shape: {data.shape}")

    # Get unique SCAT IDs and create mapping
    unique_scat_ids = sorted(data['scat_id'].unique())
    num_sites = len(unique_scat_ids)
    site_mapping = {site_id: idx for idx, site_id in enumerate(unique_scat_ids)}

    print(f"Processing data for {num_sites} unique SCAT IDs")

    # Create sequences with site IDs
    X_sequences = []
    site_indices = []
    y_values = []

    for site_id in unique_scat_ids:
        site_data = data[data['scat_id'] == site_id]
        flow_values = site_data['flow'].values

        for i in range(len(flow_values) - lag):
            X_sequences.append(flow_values[i:i+lag])
            site_indices.append(site_mapping[site_id])
            y_values.append(flow_values[i+lag])

    # Convert to numpy arrays
    X_time_series = np.array(X_sequences)
    X_site_indices = np.array(site_indices)
    y = np.array(y_values)

    # Normalize the time series data
    # For simplicity, we'll use min-max scaling to [0, 1]
    X_max = np.max(X_time_series)
    X_min = np.min(X_time_series)
    X_time_series = (X_time_series - X_min) / (X_max - X_min)

    # Normalize the target values using the same scaling
    y = (y - X_min) / (X_max - X_min)

    # Split into train/test (80/20)
    # We'll ensure the split preserves the distribution of sites
    indices = np.arange(len(X_time_series))
    np.random.shuffle(indices)
    train_size = int(len(indices) * 0.8)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train_time_series = X_time_series[train_indices]
    X_train_site_indices = X_site_indices[train_indices]
    y_train = y[train_indices]

    X_test_time_series = X_time_series[test_indices]
    X_test_site_indices = X_site_indices[test_indices]
    y_test = y[test_indices]

    return (X_train_time_series, X_train_site_indices), y_train, \
           (X_test_time_series, X_test_site_indices), y_test, \
           num_sites, site_mapping

def get_lstm_with_embedding(input_shape, num_sites, embedding_dim=8):
    """
    Create an LSTM model with site embeddings.

    # Arguments
        input_shape: tuple, shape of time series input.
        num_sites: int, number of unique SCAT sites.
        embedding_dim: int, dimension of the embedding vectors.

    # Returns
        model: Keras model with LSTM and embedding layers.
    """
    # Time series input
    time_series_input = Input(shape=input_shape)

    # Site ID input
    site_input = Input(shape=(1,))

    # Embedding layer for site IDs
    site_embedding = Embedding(input_dim=num_sites, output_dim=embedding_dim)(site_input)
    site_embedding = Flatten()(site_embedding)

    # LSTM layers
    lstm_output = LSTM(64, return_sequences=True)(time_series_input)
    lstm_output = LSTM(64)(lstm_output)

    # Combine LSTM output with site embedding
    combined = Concatenate()([lstm_output, site_embedding])

    # Output layers
    dense1 = Dense(32, activation='relu')(combined)
    output = Dense(1)(dense1)

    # Create model
    model = Model(inputs=[time_series_input, site_input], outputs=output)

    return model

def get_gru_with_embedding(input_shape, num_sites, embedding_dim=8):
    """
    Create a GRU model with site embeddings.

    # Arguments
        input_shape: tuple, shape of time series input.
        num_sites: int, number of unique SCAT sites.
        embedding_dim: int, dimension of the embedding vectors.

    # Returns
        model: Keras model with GRU and embedding layers.
    """
    # Time series input
    time_series_input = Input(shape=input_shape)

    # Site ID input
    site_input = Input(shape=(1,))

    # Embedding layer for site IDs
    site_embedding = Embedding(input_dim=num_sites, output_dim=embedding_dim)(site_input)
    site_embedding = Flatten()(site_embedding)

    # GRU layers
    gru_output = GRU(64, return_sequences=True)(time_series_input)
    gru_output = GRU(64)(gru_output)

    # Combine GRU output with site embedding
    combined = Concatenate()([gru_output, site_embedding])

    # Output layers
    dense1 = Dense(32, activation='relu')(combined)
    output = Dense(1)(dense1)

    # Create model
    model = Model(inputs=[time_series_input, site_input], outputs=output)

    return model


def get_saes_with_embedding(input_shapes, num_sites, embedding_dim=8):
    """
    Create SAES models with site embeddings that's compatible with the original model structure.

    # Arguments
        input_shapes: tuple, (time_series_shape, site_shape).
        num_sites: int, number of unique SCAT sites.
        embedding_dim: int, dimension of the embedding vectors.

    # Returns
        dict: Contains autoencoder models and final model.
    """
    time_series_shape = input_shapes[0]
    input_dim = time_series_shape[1]

    # Define the autoencoder structures (similar to original model)
    # First autoencoder: input_dim -> 400 -> input_dim
    autoencoder1_input = Input(shape=(input_dim,))
    encoded1 = Dense(400, activation='relu', name='hidden')(autoencoder1_input)
    decoded1 = Dense(input_dim)(encoded1)
    autoencoder1 = Model(inputs=autoencoder1_input, outputs=decoded1)

    # Second autoencoder: 400 -> 400 -> 400
    autoencoder2_input = Input(shape=(400,))
    encoded2 = Dense(400, activation='relu', name='hidden')(autoencoder2_input)
    decoded2 = Dense(400)(encoded2)
    autoencoder2 = Model(inputs=autoencoder2_input, outputs=decoded2)

    # Third autoencoder: 400 -> 400 -> 400
    autoencoder3_input = Input(shape=(400,))
    encoded3 = Dense(400, activation='relu', name='hidden')(autoencoder3_input)
    decoded3 = Dense(400)(encoded3)
    autoencoder3 = Model(inputs=autoencoder3_input, outputs=decoded3)

    # Now create the final model that includes site embeddings
    # Time series input
    time_series_input = Input(shape=(input_dim,))

    # Site ID input
    site_input = Input(shape=(1,))

    # Embedding layer for site IDs
    site_embedding = Embedding(input_dim=num_sites, output_dim=embedding_dim)(site_input)
    site_embedding = Flatten()(site_embedding)

    # Create stacked encoder layers (will be initialized with pre-trained weights)
    hidden1 = Dense(400, activation='relu', name='hidden1')(time_series_input)
    hidden2 = Dense(400, activation='relu', name='hidden2')(hidden1)
    hidden3 = Dense(400, activation='relu', name='hidden3')(hidden2)

    # Combine with site embedding
    combined = Concatenate()([hidden3, site_embedding])

    # Final output layers
    dense1 = Dense(32, activation='relu')(combined)
    output = Dense(1)(dense1)

    # Create final model
    final_model = Model(inputs=[time_series_input, site_input], outputs=output)

    return {
        'autoencoders': [autoencoder1, autoencoder2, autoencoder3],
        'final_model': final_model
    }

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train (lstm, gru, saes).")
    parser.add_argument(
        "--multi_site",
        action="store_true",
        help="Enable multi-site prediction mode."
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=8,
        help="Dimension for site embedding vectors."
    )
    args = parser.parse_args()

    print(f"Starting training process for {args.model} model...")
    print(f"Multi-site mode: {args.multi_site}")

    lag = 12
    config = {"batch": 256, "epochs": 100}
    file1 = 'TrainingDataAdaptedOutput.csv'

    info = {
        "model": f"{args.model}_multi" if args.multi_site else args.model,
    }

    if args.multi_site:
        print("Processing data for multi-site prediction...")
        X_train, y_train, X_test, y_test, num_sites, site_mapping = process_multi_site_data(file1, lag)

        print(f"Identified {num_sites} unique SCAT sites")
        print(f"Training data shapes: X_time_series={X_train[0].shape}, X_site_indices={X_train[1].shape}, y={y_train.shape}")

        embedding_dim = args.embedding_dim
        print(f"Using embedding dimension: {embedding_dim}")

        if args.model == 'lstm':
            print("Preparing LSTM model with site embeddings...")
            # Reshape for LSTM - [samples, timesteps, features]
            X_train_time_series = np.reshape(X_train[0], (X_train[0].shape[0], X_train[0].shape[1], 1))
            X_test_time_series = np.reshape(X_test[0], (X_test[0].shape[0], X_test[0].shape[1], 1))

            m = get_lstm_with_embedding((lag, 1), num_sites, embedding_dim)
            train_model(m, (X_train_time_series, X_train[1]), y_train, info, config)

        elif args.model == 'gru':
            print("Preparing GRU model with site embeddings...")
            # Reshape for GRU - [samples, timesteps, features]
            X_train_time_series = np.reshape(X_train[0], (X_train[0].shape[0], X_train[0].shape[1], 1))
            X_test_time_series = np.reshape(X_test[0], (X_test[0].shape[0], X_test[0].shape[1], 1))

            m = get_gru_with_embedding((lag, 1), num_sites, embedding_dim)
            train_model(m, (X_train_time_series, X_train[1]), y_train, info, config)

        elif args.model == 'saes':
            print("Preparing SAES model with site embeddings...")
            # num_sites comes from the data processing function's return value
            models = get_saes_with_embedding((X_train[0].shape, X_train[1].shape), num_sites, args.embedding_dim)
            train_saes(models, X_train, y_train, info, config)

        else:
            print(f"Unknown model type: {args.model}")

    else:
        # Original single-site code path
        print("Loading and processing data for single site...")
        try:
            X_train, y_train, _, _, _ = process_data(file1, lag)
            print("Data loaded successfully")
            print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return

        if args.model == 'lstm':
            print("Preparing LSTM model...")
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_lstm([12, 64, 64, 1])
            train_model(m, (X_train, None), y_train, info, config)
        elif args.model == 'gru':
            print("Preparing GRU model...")
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_gru([12, 64, 64, 1])
            train_model(m, (X_train, None), y_train, info, config)
        elif args.model == 'saes':
            print("Preparing SAES model...")
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
            m = model.get_saes([12, 400, 400, 400, 1])
            train_saes(m, (X_train, None), y_train, info, config)
        else:
            print(f"Unknown model type: {info.model}")

if __name__ == '__main__':
    main(sys.argv)