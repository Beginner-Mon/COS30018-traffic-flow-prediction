import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import subprocess
import os

# Read data function (from data.py)
def read_data(file, scat_no):
    '''Extracts training & testing data based on SCATS number.'''
    df1 = pd.read_excel(file, sheet_name="Data", header=0, skiprows=1)
    array = [f'V{i:02}' for i in range(96)]

    # Filter by SCATS number
    df1 = df1[df1["SCATS Number"] == scat_no][["SCATS Number", "Date"] + array]

    # Check if SCATS number exists in the dataset
    if df1.empty:
        raise ValueError(f"SCATS number {scat_no} not found in the dataset.")

    df1["Date"] = pd.to_datetime(df1['Date'])

    # Filter by date range
    training_set = df1[(df1['Date'] >= '2006-10-01') & (df1['Date'] < '2006-10-26')]
    testing_set = df1[(df1['Date'] >= '2006-10-26') & (df1['Date'] < '2006-11-01')]

    # Check if training/testing sets are empty
    if training_set.empty:
        raise ValueError("Training set is empty. Check the date range and data availability.")
    if testing_set.empty:
        raise ValueError("Testing set is empty. Check the date range and data availability.")

    # Convert to numpy arrays
    training_set = np.concatenate(training_set[array].values)
    testing_set = np.concatenate(testing_set[array].values)

    return training_set, testing_set

# Load SCATS data from Excel
def get_scats_data(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name="Data", header=0, skiprows=1)
        scats_data = df[["SCATS Number"]].drop_duplicates()
        return scats_data
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load SCATS data: {str(e)}")
        return pd.DataFrame(columns=["SCATS Number"])

# Train model
def train_model():
    selected_model = model_var.get()
    selected_scats = scats_var.get()
    print(selected_scats)

    

    # Run model training
    command = f"python train.py --model {selected_model}"
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        output = result.stdout

        # Extract Lag, Batches, and Epochs from output
        lag, batches, epochs = extract_training_info(output)

        messagebox.showinfo("Success", f"Model training completed!\n\n"
                                       f"Lag: {lag}\n"
                                       f"Batches: {batches}\n"
                                       f"Epochs: {epochs}")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Training failed:\n{e.stderr}")

# Extract training info from command output
def extract_training_info(output):
    lines = output.split("\n")
    lag, batches, epochs = "Unknown", "Unknown", "Unknown"

    for line in lines:
        if "Lag:" in line:
            lag = line.split("Lag:")[-1].strip()
        elif "Batches:" in line:
            batches = line.split("Batches:")[-1].strip()
        elif "Epochs:" in line:
            epochs = line.split("Epochs:")[-1].strip()

    return lag, batches, epochs

# Create GUI
root = tk.Tk()
root.title("Traffic Flow Prediction")

# Load SCATS data
scats_data = get_scats_data("Scats2006.xls")
scats_numbers = scats_data["SCATS Number"].unique().tolist()

# SCATS Number Selection
ttk.Label(root, text="Select SCATS Number:").grid(row=0, column=0, padx=10, pady=5)
scats_var = tk.StringVar()
scats_dropdown = ttk.Combobox(root, textvariable=scats_var, values=scats_numbers)
scats_dropdown.grid(row=0, column=1, padx=10, pady=5)

# Model Selection
ttk.Label(root, text="Select Model:").grid(row=1, column=0, padx=10, pady=5)
model_var = tk.StringVar()
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=["lstm", "gru", "saes"])
model_dropdown.grid(row=1, column=1, padx=10, pady=5)

# Train Button
train_button = ttk.Button(root, text="Train Model", command=train_model)
train_button.grid(row=2, column=0, columnspan=2, pady=10)

root.mainloop()