import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def read_data(file,scat_no):
    '''this program separates the xls file into training set and testing set'''
    df1 = pd.read_excel(file, sheet_name="Data", header=0, skiprows=1)
    array = [f'V{i:02}' for i in range(96)]

    df1 = df1[df1["SCATS Number"] == scat_no][ ["SCATS Number","Date", "Location"] + array]
   
    df1["Date"] = pd.to_datetime(df1['Date'])

 
    

    
    
    training_set = df1[(df1['Date'] >= '2006-10-01') & (df1['Date'] < '2006-10-26')]
    testing_set = df1[(df1['Date'] >= '2006-10-26') & (df1['Date'] <'2006-11-01')]
   
    # df1 = np.concatenate(df1[array].values)
    training_set = np.concatenate(training_set[array].values)
    testing_set = np.concatenate(testing_set[array].values)

    return training_set, testing_set

def read_generalised_data(file):
    df1 = pd.read_excel(file, sheet_name="Data", header=0, skiprows=1)
    array = [f'V{i:02}' for i in range(96)]

    df1 = df1[ ["SCATS Number","Date", "Location"] + array]
   
    df1["Date"] = pd.to_datetime(df1['Date'])
    
    training_set = df1[(df1['Date'] >= '2006-10-01') & (df1['Date'] < '2006-10-26')]
    testing_set = df1[(df1['Date'] >= '2006-10-26') & (df1['Date'] <'2006-11-01')]
   
    # df1 = np.concatenate(df1[array].values)
    training_set = np.concatenate(training_set[array].values)
    testing_set = np.concatenate(testing_set[array].values)

    return training_set, testing_set


def process_data(file, lags):
    df1, df2 = read_generalised_data(file)
    
    
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

    print(process_data("Scats2006.xls",12))
