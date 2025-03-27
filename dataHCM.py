import numpy as np
import pandas as pd

def read_data(file):
    '''This function reads the dataset and splits it into training and testing sets based on the date range.'''
    # Read the dataset (assuming it's a CSV file; adjust if it's Excel)
    df = pd.read_csv(file)  # Use pd.read_excel(file) if your data is in Excel format

    # Convert the 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Define the training and testing date ranges
    training_set = df[(df['date'] >= '2020-07-01') & (df['date'] <= '2021-01-31')]
    testing_set = df[(df['date'] >= '2021-02-01') & (df['date'] <= '2021-12-31')]

    # Extract the 'flow_prox' column as the target variable
    training_flow = training_set['flow_proxy'].values
    testing_flow = testing_set['flow_proxy'].values

    return training_flow, testing_flow

def process_data(file, lags):
    '''This function processes the data by creating lagged sequences for training and testing.'''
    # Read the training and testing data
    df1, df2 = read_data(file)

    # Since the data is already scaled, use it directly
    flow1 = df1  # Training flow (already scaled)
    flow2 = df2  # Testing flow (already scaled)

    # Create lagged sequences for training and testing
    train, test = [], []
    for i in range(lags, len(flow1)):
        train.append(flow1[i-lags:i+1])  # Include the current value in the sequence
    for i in range(lags, len(flow2)):
        test.append(flow2[i-lags:i+1])

    # Convert to numpy arrays
    train = np.array(train)
    test = np.array(test)

    # Shuffle the training data to avoid bias in time-series order
    np.random.shuffle(train)

    # Split into X (features) and y (target)
    X_train = train[:, :-1]  # All but the last column (lagged values)
    y_train = train[:, -1]   # The last column (target value)
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # Example usage
    file_path = "true final.csv"  # Replace with the path to your dataset file
    lags = 12  # Number of lags (can be adjusted based on your needs)
    print(process_data("true final.csv",12))