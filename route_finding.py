import pandas as pd
import networkx as nx
from geopy.distance import geodesic
from itertools import islice
from keras.api.models import load_model
import numpy as np

# Constants
SPEED_LIMIT = 60  # km/h
INTERSECTION_DELAY = 30 / 3600  # 30 seconds converted to hours
CAPACITY = 500  # Assumed capacity in vehicles

# Load SCAT data from CSV
scat_data = pd.read_csv('traffic_network2.csv')
# Build an undirected graph
G = nx.Graph()
for _, row in scat_data.iterrows():
    scat = str(row['SCATS Number'])
    lat = row['Latitude']
    lon = row['Longitude']
    neighbors = row['Neighbours'].split(';')
    G.add_node(scat, pos=(lat, lon))
    for neighbor in neighbors:
        if neighbor in G.nodes:
            dist = geodesic((lat, lon), G.nodes[neighbor]['pos']).km
            G.add_edge(scat, neighbor, distance=dist)


# Function to calculate edge travel time
def calculate_travel_time(predicted_flow_A, distance):
    base_time = distance / SPEED_LIMIT
    traffic_delay = (predicted_flow_A / CAPACITY) * (30 / 3600)
    return base_time + traffic_delay


# Function to compute total path cost including intersection delays
def compute_total_cost(path, predictions):
    edge_sum = sum(calculate_travel_time(predictions.get(path[i], 0), G[path[i]][path[i + 1]]['distance'])
                   for i in range(len(path) - 1))
    intermediate_nodes = len(path) - 2 if len(path) > 2 else 0
    total_cost = edge_sum + intermediate_nodes * INTERSECTION_DELAY
    return total_cost


# Load trained models
lstm_model = load_model('model/lstm_multi_multi_site.keras')
gru_model = load_model('model/gru_multi_multi_site.keras')
saes_model = load_model('model/saes_multi_multi_site.keras')

# Load historical data without headers
historical_data = pd.read_csv('TrainingDataAdaptedOutput.csv', header=None)
historical_data.columns = ['timestamp', 'flow', 'C', 'D', 'scat_id']

# Convert 'timestamp' to datetime
historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], format='%d/%m/%Y %H:%M')

# Define SCAT mapping from 'scat_id'
scat_ids = historical_data['scat_id'].unique()
scat_mapping = {scat: idx for idx, scat in enumerate(scat_ids)}

# Normalization parameters (replace with actual values if different)
X_min = 0
X_max = 1000


# Function to prepare input data for prediction
def prepare_input_for_prediction(historical_data, scat_mapping, lag=12, X_min=0, X_max=1000):
    latest_time = historical_data['timestamp'].max()
    # Adjust for 15-minute intervals
    recent_data = historical_data[historical_data['timestamp'] >= latest_time - pd.Timedelta(minutes=15 * (lag - 1))]
    scat_ids = list(scat_mapping.keys())
    X_time_series = []
    X_site_indices = []
    for scat in scat_ids:
        scat_data = recent_data[recent_data['scat_id'] == scat]
        scat_flows = scat_data.sort_values('timestamp')['flow'].values[-lag:]
        if len(scat_flows) == lag:
            scat_flows_normalized = (scat_flows - X_min) / (X_max - X_min)
            X_time_series.append(scat_flows_normalized)
            X_site_indices.append(scat_mapping[scat])
    return np.array(X_time_series), np.array(X_site_indices)


# Function to get model prediction
def get_model_prediction(model, X_time_series, X_site_indices, X_min, X_max):
    if 'lstm' in model.name.lower() or 'gru' in model.name.lower():
        X_time_series_reshaped = X_time_series.reshape(X_time_series.shape[0], X_time_series.shape[1], 1)
    else:
        X_time_series_reshaped = X_time_series
    pred_normalized = model.predict([X_time_series_reshaped, X_site_indices], verbose=0)
    pred = pred_normalized * (X_max - X_min) + X_min
    return pred.flatten()


# Function to get ensemble predictions
def get_ensemble_predictions(lstm_model, gru_model, saes_model, X_time_series, X_site_indices,
                             weights=[1 / 4, 1 / 4, 1 / 2], X_min=0, X_max=1000):
    lstm_pred = get_model_prediction(lstm_model, X_time_series, X_site_indices, X_min, X_max)
    gru_pred = get_model_prediction(gru_model, X_time_series, X_site_indices, X_min, X_max)
    saes_pred = get_model_prediction(saes_model, X_time_series, X_site_indices, X_min, X_max)
    ensemble_pred = weights[0] * lstm_pred + weights[1] * gru_pred + weights[2] * saes_pred
    return ensemble_pred


# Get origin and destination from user input
origin = input("Enter origin SCAT number: ").strip()
destination = input("Enter destination SCAT number: ").strip()

# Validate user input
if origin not in G or destination not in G:
    print("One or both SCAT numbers do not exist in the graph.")
elif origin == destination:
    print("Origin and destination are the same.")
else:
    # Prepare input data
    X_time_series, X_site_indices = prepare_input_for_prediction(historical_data, scat_mapping, X_min=X_min,
                                                                 X_max=X_max)

    # Get ensemble predictions
    ensemble_pred = get_ensemble_predictions(lstm_model, gru_model, saes_model, X_time_series, X_site_indices,
                                             X_min=X_min, X_max=X_max)

    # Create predictions dictionary
    predictions = {scat: ensemble_pred[i] for i, scat in enumerate(scat_ids) if i < len(ensemble_pred)}

    # Assign edge weights based on predicted flows and distances
    for u, v in G.edges:
        flow_u = predictions.get(u, 0)
        dist = G[u][v]['distance']
        travel_time = calculate_travel_time(flow_u, dist)
        G[u][v]['weight'] = travel_time

    # Find up to 100 shortest paths based on edge weights
    try:
        paths = list(islice(nx.shortest_simple_paths(G, origin, destination, weight='weight'), 100))
    except nx.NetworkXNoPath:
        print("No path exists between the origin and destination.")
        paths = []

    # Calculate total travel time for each path
    path_costs = []
    for path in paths:
        cost = compute_total_cost(path, predictions)
        path_costs.append((path, cost))

    # Sort by total cost and select up to 5 routes
    sorted_paths = sorted(path_costs, key=lambda x: x[1])[:5]

    # Display the results
    print("\nTop 5 Routes:")
    for i, (path, cost) in enumerate(sorted_paths, 1):
        total_time_minutes = cost * 60
        print(f"Route {i}: {' -> '.join(path)} (Travel Time: {total_time_minutes:.2f} minutes)")