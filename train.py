"""
Train a neural network model with support for multiple SCAT sites using embeddings.
Supports LSTM, GRU, and SAES models for traffic flow prediction across 41 SCAT IDs.
"""

import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api.layers import LSTM, GRU, Dropout
from keras import Model
from keras.api.layers import Input, Embedding, Flatten, Concatenate, Dense
from keras.api.callbacks import EarlyStopping
from keras.api.losses import MeanSquaredError
from keras.api.optimizers import Adam
warnings.filterwarnings("ignore")

# --- Training Functions ---

def train_model(model, X_train, y_train, X_test, y_test, name, config):
    """
    Train a single model and evaluate it on the test set.

    Arguments:
        model: Keras Model, the neural network model to train.
        X_train: tuple (time_series_data, site_ids), input data for training.
        y_train: ndarray, target data for training.
        X_test: tuple (time_series_data, site_ids), input data for testing.
        y_test: ndarray, target data for testing.
        name: dict, contains the model name (e.g., {'model': 'lstm_multi'}).
        config: dict, training parameters (batch size, epochs).
    """
    print(f"Compiling {name['model']} model...")
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=['mape'])

    print(f"Starting training for {name['model']}...")
    time_series_data, site_ids = X_train
    print(f"Time series data shape: {time_series_data.shape}")
    print(f"Site IDs shape: {site_ids.shape}")
    print(f"Output shape: {y_train.shape}")

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    hist = model.fit(
        [time_series_data, site_ids], y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_mape = model.evaluate([X_test[0], X_test[1]], y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.6f}, Test MAPE: {test_mape:.2f}%")

    print(f"Saving {name['model']} model...")
    model.save(f"model/{name['model']}.keras")
    pd.DataFrame(hist.history).to_csv(f"model/{name['model']}_loss.csv", encoding='utf-8', index=False)
    print(f"Training completed for {name['model']}")

def train_saes(models, X_train, y_train, X_test, y_test, name, config):
    """
    Train the SAES model with site embeddings and evaluate it.

    Arguments:
        models: dict, contains autoencoder models and the final model.
        X_train: tuple (time_series_data, site_ids), input data for training.
        y_train: ndarray, target data for training.
        X_test: tuple (time_series_data, site_ids), input data for testing.
        y_test: ndarray, target data for testing.
        name: dict, contains the model name (e.g., {'model': 'saes_multi'}).
        config: dict, training parameters (batch size, epochs).
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
        m.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=['mape'])

        print(f"Training autoencoder {i+1}")
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        m.fit(temp, temp,
              batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05,
              callbacks=[early_stopping],
              verbose=1)

        autoencoder_models[i] = m

    # Transfer learned weights to the final model
    print("Transferring weights to final model...")
    for i in range(len(autoencoder_models)):
        weights = autoencoder_models[i].get_layer('hidden').get_weights()
        final_model.get_layer(f'hidden{i+1}').set_weights(weights)

    # Fine-tune the final model
    print("Fine-tuning the final SAES model...")
    final_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=['mape'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    hist = final_model.fit(
        [time_series_data, site_ids], y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_mape = final_model.evaluate([X_test[0], X_test[1]], y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.6f}, Test MAPE: {test_mape:.2f}%")

    print(f"Saving {name['model']} model...")
    final_model.save(f"model/{name['model']}.keras")
    pd.DataFrame(hist.history).to_csv(f"model/{name['model']}_loss.csv", index=False)

# --- Data Processing ---

def process_multi_site_data(file_path, lag):
    """
    Process data for multiple SCAT sites with proper train/test normalization.

    Arguments:
        file_path: str, path to the data file.
        lag: int, length of historical data to use for sequences.

    Returns:
        X_train: tuple (time_series_data, site_ids) for training.
        y_train: ndarray, target values for training.
        X_test: tuple (time_series_data, site_ids) for testing.
        y_test: ndarray, target values for testing.
        num_sites: int, number of unique SCAT sites.
        site_mapping: dict, mapping of SCAT IDs to indices.
    """
    print(f"Loading data from {file_path}...")
    try:
        data = pd.read_csv(file_path)
        data.columns = ["timestamp", "flow", "day", "day_num", "scat_id"]
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        raise Exception(f"Could not load data from {file_path}")

    print(f"Data loaded with shape: {data.shape}")

    unique_scat_ids = sorted(data['scat_id'].unique())
    num_sites = len(unique_scat_ids)
    site_mapping = {site_id: idx for idx, site_id in enumerate(unique_scat_ids)}

    print(f"Processing data for {num_sites} unique SCAT IDs")

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

    X_time_series = np.array(X_sequences)
    X_site_indices = np.array(site_indices)
    y = np.array(y_values)

    # Split into train/test before normalization
    indices = np.arange(len(X_time_series))
    np.random.shuffle(indices)
    train_size = int(len(indices) * 0.8)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train_ts = X_time_series[train_indices]
    X_train_si = X_site_indices[train_indices]
    y_train = y[train_indices]

    X_test_ts = X_time_series[test_indices]
    X_test_si = X_site_indices[test_indices]
    y_test = y[test_indices]

    # Normalize based on training data only
    X_min = np.min(X_train_ts)
    X_max = np.max(X_train_ts)
    X_train_ts = (X_train_ts - X_min) / (X_max - X_min)
    X_test_ts = (X_test_ts - X_min) / (X_max - X_min)
    y_train = (y_train - X_min) / (X_max - X_min)
    y_test = (y_test - X_min) / (X_max - X_min)

    return (X_train_ts, X_train_si), y_train, (X_test_ts, X_test_si), y_test, num_sites, site_mapping

# --- Model Definitions ---

def get_lstm_with_embedding(input_shape, num_sites, embedding_dim=8):
    """Create an LSTM model with site embeddings and dropout."""
    time_series_input = Input(shape=input_shape)
    site_input = Input(shape=(1,))

    site_embedding = Embedding(input_dim=num_sites, output_dim=embedding_dim)(site_input)
    site_embedding = Flatten()(site_embedding)

    lstm_output = LSTM(64, return_sequences=True)(time_series_input)
    lstm_output = Dropout(0.2)(lstm_output)  # Add dropout for regularization
    lstm_output = LSTM(64)(lstm_output)
    lstm_output = Dropout(0.2)(lstm_output)

    combined = Concatenate()([lstm_output, site_embedding])
    dense1 = Dense(32, activation='relu')(combined)
    output = Dense(1)(dense1)

    model = Model(inputs=[time_series_input, site_input], outputs=output)
    return model

def get_gru_with_embedding(input_shape, num_sites, embedding_dim=8):
    """Create a GRU model with site embeddings and dropout."""
    time_series_input = Input(shape=input_shape)
    site_input = Input(shape=(1,))

    site_embedding = Embedding(input_dim=num_sites, output_dim=embedding_dim)(site_input)
    site_embedding = Flatten()(site_embedding)

    gru_output = GRU(64, return_sequences=True)(time_series_input)
    gru_output = Dropout(0.2)(gru_output)  # Add dropout for regularization
    gru_output = GRU(64)(gru_output)
    gru_output = Dropout(0.2)(gru_output)

    combined = Concatenate()([gru_output, site_embedding])
    dense1 = Dense(32, activation='relu')(combined)
    output = Dense(1)(dense1)

    model = Model(inputs=[time_series_input, site_input], outputs=output)
    return model

def get_saes_with_embedding(input_shapes, num_sites, embedding_dim=8):
    """Create SAES models with site embeddings and dropout."""
    time_series_shape = input_shapes[0]
    input_dim = time_series_shape[1]

    # Define autoencoders
    autoencoder1_input = Input(shape=(input_dim,))
    encoded1 = Dense(400, activation='relu', name='hidden')(autoencoder1_input)
    decoded1 = Dense(input_dim)(encoded1)
    autoencoder1 = Model(inputs=autoencoder1_input, outputs=decoded1)

    autoencoder2_input = Input(shape=(400,))
    encoded2 = Dense(400, activation='relu', name='hidden')(autoencoder2_input)
    decoded2 = Dense(400)(encoded2)
    autoencoder2 = Model(inputs=autoencoder2_input, outputs=decoded2)

    autoencoder3_input = Input(shape=(400,))
    encoded3 = Dense(400, activation='relu', name='hidden')(autoencoder3_input)
    decoded3 = Dense(400)(encoded3)
    autoencoder3 = Model(inputs=autoencoder3_input, outputs=decoded3)

    # Final model with embeddings
    time_series_input = Input(shape=(input_dim,))
    site_input = Input(shape=(1,))

    site_embedding = Embedding(input_dim=num_sites, output_dim=embedding_dim)(site_input)
    site_embedding = Flatten()(site_embedding)

    hidden1 = Dense(400, activation='relu', name='hidden1')(time_series_input)
    hidden2 = Dense(400, activation='relu', name='hidden2')(hidden1)
    hidden3 = Dense(400, activation='relu', name='hidden3')(hidden2)
    hidden3 = Dropout(0.2)(hidden3)  # Add dropout for regularization

    combined = Concatenate()([hidden3, site_embedding])
    dense1 = Dense(32, activation='relu')(combined)
    output = Dense(1)(dense1)

    final_model = Model(inputs=[time_series_input, site_input], outputs=output)

    return {
        'autoencoders': [autoencoder1, autoencoder2, autoencoder3],
        'final_model': final_model
    }

# --- Main Execution ---

def main(argv):
    parser = argparse.ArgumentParser(description="Train a neural network model for traffic flow prediction.")
    parser.add_argument("--model", default="lstm", help="Model to train (lstm, gru, saes).")
    parser.add_argument("--multi_site", action="store_true", help="Enable multi-site prediction mode.")
    parser.add_argument("--embedding_dim", type=int, default=8, help="Dimension for site embedding vectors.")
    args = parser.parse_args()

    print(f"Starting training process for {args.model} model...")
    print(f"Multi-site mode: {args.multi_site}")

    lag = 12  # Length of historical data sequence
    config = {"batch": 256, "epochs": 100}  # Training configuration
    file_path = 'TrainingDataAdaptedOutput.csv'  # Adjust this to your data file path

    info = {"model": f"{args.model}_multi" if args.multi_site else args.model}

    if args.multi_site:
        print("Processing data for multi-site prediction...")
        X_train, y_train, X_test, y_test, num_sites, site_mapping = process_multi_site_data(file_path, lag)

        print(f"Identified {num_sites} unique SCAT sites")
        print(f"Training data shapes: X_time_series={X_train[0].shape}, X_site_indices={X_train[1].shape}, y={y_train.shape}")

        embedding_dim = args.embedding_dim
        print(f"Using embedding dimension: {embedding_dim}")

        if args.model == 'lstm':
            print("Preparing LSTM model with site embeddings...")
            X_train_ts = np.reshape(X_train[0], (X_train[0].shape[0], X_train[0].shape[1], 1))
            X_test_ts = np.reshape(X_test[0], (X_test[0].shape[0], X_test[0].shape[1], 1))
            m = get_lstm_with_embedding((lag, 1), num_sites, embedding_dim)
            train_model(m, (X_train_ts, X_train[1]), y_train, (X_test_ts, X_test[1]), y_test, info, config)

        elif args.model == 'gru':
            print("Preparing GRU model with site embeddings...")
            X_train_ts = np.reshape(X_train[0], (X_train[0].shape[0], X_train[0].shape[1], 1))
            X_test_ts = np.reshape(X_test[0], (X_test[0].shape[0], X_test[0].shape[1], 1))
            m = get_gru_with_embedding((lag, 1), num_sites, embedding_dim)
            train_model(m, (X_train_ts, X_train[1]), y_train, (X_test_ts, X_test[1]), y_test, info, config)

        elif args.model == 'saes':
            print("Preparing SAES model with site embeddings...")
            models = get_saes_with_embedding((X_train[0].shape, X_train[1].shape), num_sites, embedding_dim)
            train_saes(models, X_train, y_train, X_test, y_test, info, config)

        else:
            print(f"Unknown model type: {args.model}")
            return

    else:
        print("Single-site mode is not implemented in this version. Use --multi_site for 41 SCAT IDs.")
        return

if __name__ == '__main__':
    main(sys.argv)