# check_data.py
# This script checks and preprocesses traffic flow data to handle NaN values.

import pandas as pd
import numpy as np

# Step 1: Load the data
def load_data(file_path):
    """
    Load the traffic flow data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    return df

# Step 2: Inspect for missing values
def inspect_missing_values(df):
    """
    Check for missing values in the DataFrame.
    
    :param df: Input DataFrame
    """
    print("\nMissing values per column:")
    print(df.isnull().sum())

# Step 3: Handle missing values
def handle_missing_values(df):
    """
    Handle missing values in the 'flow' column using interpolation and fill methods.
    
    :param df: Input DataFrame
    :return: DataFrame with missing values handled
    """
    # Ensure 'timestamp' is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
    
    # Get unique scat_ids
    scat_ids = df['scat_id'].unique()
    
    # Generate a complete timestamp range (every 15 minutes)
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    all_timestamps = pd.date_range(start=start_time, end=end_time, freq='15min')
    
    # Create a MultiIndex with all timestamp and scat_id combinations
    idx = pd.MultiIndex.from_product([all_timestamps, scat_ids], names=['timestamp', 'scat_id'])
    
    # Reindex the DataFrame to include all combinations
    df = df.set_index(['timestamp', 'scat_id']).reindex(idx).reset_index()
    
    # Impute 'flow' by scat_id using interpolation
    df['flow'] = df.groupby('scat_id')['flow'].transform(lambda x: x.interpolate(method='linear'))
    
    # Fill remaining NaNs with forward and backward fill
    df['flow'] = df.groupby('scat_id')['flow'].ffill()
    df['flow'] = df.groupby('scat_id')['flow'].bfill()
    
    # If NaNs still exist, fill with 0
    df['flow'] = df['flow'].fillna(0)
    
    print("\nMissing values handled.")
    return df

# Step 4: Derive time features
def derive_time_features(df):
    """
    Derive 'hour', 'weekend', and 'day' features from 'timestamp'.
    
    :param df: Input DataFrame
    :return: DataFrame with derived features
    """
    df['hour'] = df['timestamp'].dt.hour
    df['weekend'] = df['timestamp'].dt.dayofweek >= 5
    df['day'] = df['timestamp'].dt.dayofweek
    print("\nTime features derived.")
    return df

# Step 5: Pivot the data
def pivot_data(df):
    """
    Pivot the DataFrame into a wide format for modeling.
    
    :param df: Input DataFrame
    :return: Pivoted DataFrame
    """
    feature_df = df.pivot_table(
        index='timestamp',
        columns='scat_id',
        values=['flow', 'hour', 'weekend', 'day'],
        aggfunc='first'
    )
    # Flatten column names
    feature_df.columns = [f'{col[0]}_{col[1]}' for col in feature_df.columns]
    print("\nData pivoted successfully.")
    return feature_df

# Step 6: Final checks
def final_checks(feature_df):
    """
    Perform final checks to ensure no NaNs remain.
    
    :param feature_df: Pivoted DataFrame
    :return: Cleaned DataFrame
    """
    if feature_df.isnull().values.any():
        print("\nWarning: NaNs still present in the data.")
        feature_df = feature_df.fillna(0)
    else:
        print("\nNo NaNs found in the data.")
    return feature_df

# Main function to run all steps
def main(file_path):
    """
    Main function to run all data checking and preprocessing steps.
    
    :param file_path: Path to the CSV file
    """
    # Load data
    df = load_data(file_path)
    
    # Inspect missing values
    inspect_missing_values(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Derive time features
    df = derive_time_features(df)
    
    # Pivot the data
    feature_df = pivot_data(df)
    
    # Final checks
    feature_df = final_checks(feature_df)
    
    # Save the preprocessed data
    feature_df.to_csv('preprocessed_traffic_data.csv', index=True)
    print("\nPreprocessed data saved to 'preprocessed_traffic_data.csv'.")

# Run the script (replace 'traffic_data.csv' with your actual file path)
if __name__ == "__main__":
    main('TrainingDataAdaptedOutput.csv')