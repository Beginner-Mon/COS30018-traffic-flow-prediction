# routefinding.py
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

# Load SCAT data and build graph
scat_data = pd.read_csv('traffic_network2.csv')
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

# Load trained models and historical data
lstm_model = load_model('model/lstm_multi_multi_site.keras')
gru_model = load_model('model/gru_multi_multi_site.keras')
saes_model = load_model('model/saes_multi_multi_site.keras')

historical_data = pd.read_csv('TrainingDataAdaptedOutput.csv', header=None,skiprows=1)
historical_data.columns = ['timestamp', 'flow', 'day', 'day_num', 'scat_id', 'direction']
historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], format='%d/%m/%Y %H:%M')

scat_ids = historical_data['scat_id'].unique()
scat_mapping = {scat: idx for idx, scat in enumerate(scat_ids)}
X_min = 0
X_max = 1000
def get_all_scats_data():
    '''
    Returns a list of all SCAT data in network2.csv
    '''
    return scat_data[['SCATS Number', 'Latitude', 'Longitude', 'Site Description', 'Site Type', 'Neighbours']].to_dict('records')
# Functions
def calculate_travel_time(predicted_flow_A, distance):
    base_time = distance / SPEED_LIMIT
    traffic_delay = (predicted_flow_A / CAPACITY) * (30 / 3600)
    return base_time + traffic_delay

def compute_total_cost(path, predictions):
    edge_sum = sum(calculate_travel_time(predictions.get(path[i], 0), G[path[i]][path[i + 1]]['distance'])
                   for i in range(len(path) - 1))
    intermediate_nodes = len(path) - 2 if len(path) > 2 else 0
    total_cost = edge_sum + intermediate_nodes * INTERSECTION_DELAY
    return total_cost

def prepare_input_for_prediction(historical_data, scat_mapping, lag=12, X_min=0, X_max=1000):
    latest_time = historical_data['timestamp'].max()
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

def get_model_prediction(model, X_time_series, X_site_indices, X_min, X_max):
    if 'lstm' in model.name.lower() or 'gru' in model.name.lower():
        X_time_series_reshaped = X_time_series.reshape(X_time_series.shape[0], X_time_series.shape[1], 1)
    else:
        X_time_series_reshaped = X_time_series
    pred_normalized = model.predict([X_time_series_reshaped, X_site_indices], verbose=0)
    pred = pred_normalized * (X_max - X_min) + X_min
    return pred.flatten()

def get_ensemble_predictions(lstm_model, gru_model, saes_model, X_time_series, X_site_indices,
                             weights=[1 / 4, 1 / 4, 1 / 2], X_min=0, X_max=1000):
    lstm_pred = get_model_prediction(lstm_model, X_time_series, X_site_indices, X_min, X_max)
    gru_pred = get_model_prediction(gru_model, X_time_series, X_site_indices, X_min, X_max)
    saes_pred = get_model_prediction(saes_model, X_time_series, X_site_indices, X_min, X_max)
    ensemble_pred = weights[0] * lstm_pred + weights[1] * gru_pred + weights[2] * saes_pred
    return ensemble_pred

# Hàm tính toán tuyến đường (dùng cho API)
def find_top_routes(origin, destination):
    if origin not in G or destination not in G:
        return None, "One or both SCAT numbers do not exist in the graph"
    if origin == destination:
        return None, "Origin and destination are the same"

    X_time_series, X_site_indices = prepare_input_for_prediction(historical_data, scat_mapping, X_min=X_min, X_max=X_max)
    ensemble_pred = get_ensemble_predictions(lstm_model, gru_model, saes_model, X_time_series, X_site_indices, X_min=X_min, X_max=X_max)
    predictions = {scat: ensemble_pred[i] for i, scat in enumerate(scat_ids) if i < len(ensemble_pred)}

    for u, v in G.edges:
        flow_u = predictions.get(u, 0)
        dist = G[u][v]['distance']
        travel_time = calculate_travel_time(flow_u, dist)
        G[u][v]['weight'] = travel_time

    try:
        paths = list(islice(nx.shortest_simple_paths(G, origin, destination, weight='weight'), 100))
    except nx.NetworkXNoPath:
        return None, "No path exists between the origin and destination"

    path_costs = []
    for path in paths:
        cost = compute_total_cost(path, predictions)
        path_costs.append((path, cost))

    sorted_paths = sorted(path_costs, key=lambda x: x[1])[:5]

    routes = []
    for i, (path, cost) in enumerate(sorted_paths, 1):
        total_time_minutes = cost * 60
        routes.append({
            'route_number': i,
            'path': path,
            'travel_time_minutes': round(total_time_minutes, 2)
        })

    return routes, None

# Hàm main để chạy logic gốc
def main():
    origin = input("Enter origin SCAT number: ").strip()
    destination = input("Enter destination SCAT number: ").strip()

    routes, error = find_top_routes(origin, destination)

    if error:
        print(error)
    else:
        print("\nTop 5 Routes:")
        for route in routes:
            print(f"Route {route['route_number']}: {' -> '.join(route['path'])} (Travel Time: {route['travel_time_minutes']:.2f} minutes)")

# Chạy main nếu file được chạy trực tiếp
if __name__ == '__main__':
    main()