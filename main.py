from data import process_data
import numpy as np
import pandas as pd
import math

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

import matplotlib as mpl
import matplotlib.pyplot as plt
def plot_results(y_true, y_preds, names):
    d = '2006-6-1 00:00'
    x = pd.date_range(d, periods = 288, freq = "5min")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label = 'True Data')
    ax.plot(x,y_preds[0].flatten(), label = names[0])

    plt.legend()
    plt.grid(True)
    plt.xlabel("Time of Day")
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()

def main():
    lstm = load_model("./lstm.keras")
    gru = load_model("./gru.keras")
    models = [lstm, gru]
    names = ["LSTM", "GRU", "SAEs"]
    
    lag = 12
    file1 = "./Scats Data October 2006.xls"
    file2 = "./Scats Data October 2006.xls"
    _, _ , X_test, y_test, scaler = process_data(file1,file2, lag)
    y_test = scaler.inverse_transform(y_test.reshape(-1,1)).reshape(1,-1)[0]

    y_preds = []
    for name,model in zip(names,models):

        X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1 ))
        file = f'images/{name}.png'
        plot_model(model, to_file = file, show_shapes = True)
        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1,1)).reshape(1,-1)[0]
        y_preds.append(predicted[:288])
        print(name)

    plot_results(y_test[: 288], y_preds, names)

if __name__ == "__main__":
    main()