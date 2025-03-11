import pandas as pd
import matplotlib.pyplot as plt
from keras.api.models import load_model
from route_finding2 import prepare_data, evaluate_model

# Load training history
lstm_hist = pd.read_csv("model/lstm_multi_multi_site_loss.csv")
gru_hist = pd.read_csv("model/gru_multi_multi_site_loss.csv")
saes_hist = pd.read_csv("model/saes_multi_loss.csv")  # Now exists!

# Plot validation loss
plt.plot(lstm_hist['val_loss'], label='LSTM')
plt.plot(gru_hist['val_loss'], label='GRU')
plt.plot(saes_hist['val_loss'], label='SAES')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss (MAE)')
plt.legend()
plt.show()

# Load models
lstm = load_model("model/lstm_multi_multi_site.keras")
gru = load_model("model/gru_multi_multi_site.keras")
saes = load_model("model/saes_multi_multi_site.keras")

# Load test data
data = pd.read_csv("TrainingDataAdaptedOutput.csv")  # Load your raw data file
_, _, _, X_test, X_site_test, y_test = prepare_data(data)

# Evaluate models
lstm_metrics = evaluate_model(lstm, X_test, X_site_test, y_test)
gru_metrics = evaluate_model(gru, X_test, X_site_test, y_test)
saes_metrics = evaluate_model(saes, X_test, X_site_test, y_test)

print("LSTM Overall MAE:", lstm_metrics['Overall MAE'])
print("GRU Overall MAE:", gru_metrics['Overall MAE'])
print("SAES Overall MAE:", saes_metrics['Overall MAE'])