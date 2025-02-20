import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def process_data(train, test, lags):
    df1 = pd.read_excel(train, sheet_name="Data", header=0, skiprows=1)
    df2 = pd.read_excel(test, sheet_name="Data", header=0, skiprows=1)

    array = [f'V{i:02}' for i in range(95)]
    df1 = df1[df1["SCATS Number"] == 970][ array]
    df2 = df2[df2["SCATS Number"] == 970][ array]
    # Print the resulting DataFrame
    df1 = np.concatenate(df1[array].values)
    df2 = np.concatenate(df2[array].values)
    
    
    scaler = MinMaxScaler(feature_range=(0,1))
    flow1 = scaler.fit_transform(df1.reshape(-1,1))
    flow2 = scaler.fit_transform(df2.reshape(-1,1))
    
    train, test = [], []
    for i in range(lags, len(flow1)):   
        train.append(flow1[i-lags: i+1])
    for i in range(lags, len(flow2)):
        test.append(flow2[i-lags: i+1])

    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test, scaler


    

if __name__ == "__main__":

    process_data('./train.csv','./Scats Data October 2006.xls', 12)
