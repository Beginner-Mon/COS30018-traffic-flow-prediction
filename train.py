from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import model
import sys
import argparse

from data import process_data
def train_model(model, X_train, Y_train, name, config):
    model.compile(loss = "mean_squared_error", optimizer ="adam")

    hist = model.fit(
        X_train, Y_train, 
        epochs = config["epochs"], 
        verbose = 0,
        validation_split = 0.05, 
        batch_size = config["batch"]
        )
    model.save(f"./{name}.keras")

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(f'./model/{name} loss.csv', encoding='utf 8', index = False)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train.")
    args = parser.parse_args()

    lag = 12
    config = {"batch":256, "epochs": 2}
    file1  = "./Scats Data October 2006.xls"
    X_train , Y_train, _,_ ,_ = process_data(file1, file1, lag)
    if args.model == 'lstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_lstm([12,64,64,1])
        train_model(m, X_train, Y_train, args.model, config)
    if args.model == "gru":
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_lstm([12,64,64,1])
        train_model(m, X_train, Y_train, args.model, config)

if __name__ == "__main__":
    main(sys.argv)
