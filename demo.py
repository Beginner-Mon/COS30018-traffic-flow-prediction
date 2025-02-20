import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Generate synthetic traffic data for demonstration
def generate_synthetic_data(num_samples=1000, timesteps=10):
    # Generate random traffic flow data
    data = np.random.rand(num_samples, timesteps, 1)  # Random data
    labels = np.random.rand(num_samples, 1)  # Random labels
    return data, labels

# Load or generate your dataset
X, y = generate_synthetic_data()

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

# Make predictions
predictions = model.predict(X_test)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title('Traffic Flow Prediction')
plt.xlabel('Sample Index')
plt.ylabel('Traffic Flow')
plt.show()