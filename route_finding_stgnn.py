import pandas as pd
import numpy as np
import networkx as nx
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import StandardScaler
import math
import torch
# Replace with your actual STGNN module imports
from claude_solution import EnhancedSTGNN #, TrafficDataset, DataLoader for process multiple sequences in future

# Constants
SPEED_LIMIT = 60  # km/h
MIN_SPEED = 5     # km/h
DELAY_PER_INTERSECTION = 30  # seconds
FLOW_THRESHOLD = 1000  # Flow value indicating congestion

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points in kilometers."""
    coords_1 = np.radians([lat1, lon1])
    coords_2 = np.radians([lat2, lon2])
    distance = haversine_distances([coords_1, coords_2]) * 6371  # Earth radius in km
    return distance[0, 1]

def build_graph(spatial_file):
    """Build a directed graph from the spatial data."""
    spatial_df = pd.read_csv(spatial_file)
    G = nx.DiGraph()
    for _, row in spatial_df.iterrows():
        scat = str(row['SCATS Number'])
        lat, lon = row['Latitude'], row['Longitude']
        G.add_node(scat, lat=lat, lon=lon)
        neighbors = [str(n).strip() for n in str(row['Neighbours']).split(',') if n.strip()]
        for nbr in neighbors:
            if nbr in G.nodes:
                dist = haversine_distance(lat, lon, G.nodes[nbr]['lat'], G.nodes[nbr]['lon'])
                G.add_edge(scat, nbr, distance=dist)
    return G

def round_to_nearest_15min(dt):
    """Round a datetime to the nearest 15-minute interval."""
    minute = dt.minute
    minute_rounded = round(minute / 15) * 15
    if minute_rounded == 60:
        dt += timedelta(hours=1)
        minute_rounded = 0
    return dt.replace(minute=minute_rounded, second=0, microsecond=0)

import pandas as pd
import torch

def predict_flows(model, data_tensor, start_time, actual_timestamps, window_size=24):
    """
    Predict flows starting from start_time using historical data with timestamps.
    
    Args:
        model: Trained model (e.g., STGNN)
        data_tensor: Historical data tensor [time_steps, nodes, features]
        start_time: datetime object for prediction start
        window_size: Number of past time steps to use
    Returns:
        Predicted flows as a numpy array
    """
    # Example timestamps: 15-minute intervals starting from some date
    timestamps = pd.DatetimeIndex(actual_timestamps)

    # Find closest index using real timestamps
    time_diffs = (timestamps - start_time).to_series().abs()
    idx = time_diffs.idxmin()

    # Extract the previous window_size time steps
    if idx >= window_size:
        input_sequence = data_tensor[idx - window_size:idx, :, :]
    else:
        raise ValueError("Not enough historical data before start_time")

    # Prepare input for the model
    device = next(model.parameters()).device
    input_sequence = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        preds, _ = model(input_sequence)  # Unpack tuple (output, attention)
        
    return preds.squeeze(0).cpu().numpy()

def interpolate_flows(predicted_flows, num_intervals=3):
    """
    Interpolate flows from 15-minute to 5-minute intervals using cubic spline.
    
    Args:
        predicted_flows: Array of shape [horizon, num_nodes].
        num_intervals: Number of 5-min intervals per 15-min interval (default: 3).
    
    Returns:
        np.ndarray: Interpolated flows of shape [horizon * num_intervals, num_nodes].
    """
    horizon, num_nodes = predicted_flows.shape
    original_times = np.arange(0, horizon * 15, 15)
    interp_times = np.arange(0, horizon * 15, 5)
    cs = CubicSpline(original_times, predicted_flows, axis=0)
    interpolated_flows = cs(interp_times)
    return interpolated_flows

def calculate_speed(flow):
    """Calculate speed based on flow using the provided formula."""
    if flow > 1500:
        return MIN_SPEED
    try:
        speed = 32 + math.sqrt(1024 - (256 / 375) * flow)
        return max(min(speed, SPEED_LIMIT), MIN_SPEED)
    except ValueError:
        return MIN_SPEED

def calculate_travel_time(distance, speed, delay=DELAY_PER_INTERSECTION):
    """Calculate travel time in minutes, including intersection delay."""
    time_hours = distance / speed
    time_minutes = time_hours * 60
    return time_minutes + (delay / 60)

def yen_k_shortest_paths(G, source, target, k=5):
    """
    Find k shortest paths using Yen's algorithm (simplified).
    
    Args:
        G: NetworkX DiGraph with weights.
        source: Starting SCATS number.
        target: Destination SCATS number.
        k: Number of paths to find (default: 5).
    
    Returns:
        list: List of k shortest paths.
    """
    try:
        # Simplified: Uses NetworkX's shortest_simple_paths
        # For production, implement full Yen's algorithm or use a library
        paths = list(nx.shortest_simple_paths(G, source, target, weight='weight'))
        return paths[:min(k, len(paths))]
    except nx.NetworkXNoPath:
        print(f"No path found between {source} and {target}.")
        return []

def get_crowded_nodes(paths, interpolated_flows, scat_ids, flow_threshold=FLOW_THRESHOLD):
    """
    Identify nodes along the paths predicted to be crowded.
    
    Args:
        paths: List of paths (lists of SCATS numbers).
        interpolated_flows: Array of shape [num_intervals, num_nodes].
        scat_ids: List mapping node indices to SCATS numbers.
        flow_threshold: Flow value indicating congestion.
    
    Returns:
        dict: Mapping of crowded SCATS numbers to list of time offsets (in minutes).
    """
    crowded_info = {}
    num_intervals = interpolated_flows.shape[0]
    scat_to_idx = {scat: idx for idx, scat in enumerate(scat_ids)}
    
    for t in range(num_intervals):
        for path in paths:
            for node in path:
                if node in scat_to_idx:
                    flow = interpolated_flows[t, scat_to_idx[node]]
                    if flow > flow_threshold:
                        if node not in crowded_info:
                            crowded_info[node] = []
                        crowded_info[node].append(t * 5)  # Time in minutes
    return crowded_info

def main(start_scat, dest_scat, current_time_str, spatial_file, temporal_file, model_path, output_dir):
    """
    Main function to find routes, predict travel times, and identify crowded nodes.
    
    Args:
        start_scat: Starting SCATS number (str).
        dest_scat: Destination SCATS number (str).
        current_time_str: Current time in 'YYYY-MM-DD HH:MM:SS' format.
        spatial_file: Path to spatial data CSV (e.g., 'traffic_network2.csv').
        temporal_file: Path to temporal data CSV.
        model_path: Path to pre-trained STGNN model file.
        output_dir: Directory to save results.
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build graph
    G = build_graph(spatial_file)

    # Load and preprocess temporal data (placeholder function)
    # Replace with your actual preprocessing function
    # Expected return: adjacency matrix, data tensor, SCATS IDs, coordinates
    def preprocess_traffic_data(spatial_file, temporal_file):
        """
        Preprocess spatial and temporal traffic data into a tensor.

        Args:
            spatial_file: Path to CSV with spatial data (SCATS Number, coordinates).
            temporal_file: Path to CSV with temporal data (timestamp, scat_id, flow).

        Returns:
            adj_matrix: Adjacency matrix for the graph.
            data_tensor: Tensor of shape [time, nodes, features].
            scat_ids: List of SCATS node IDs.
            coordinates: Coordinates of nodes.
            scaler: Fitted scaler for inverse transformation.
        """
        # Load spatial data
        spatial_df = pd.read_csv(spatial_file)
        scat_ids = sorted(spatial_df['SCATS Number'].unique())
        num_nodes = len(scat_ids)
        coordinates = spatial_df[['Latitude', 'Longitude']].values  # Adjust column names as needed

        # Create adjacency matrix (example: based on connectivity or distance)
        adj_matrix = np.zeros((num_nodes, num_nodes))  # Update based on your data

        # Load temporal data
        temporal_df = pd.read_csv(temporal_file)
        
        # Parse timestamps with the correct day-first format
        temporal_df['timestamp'] = pd.to_datetime(temporal_df['timestamp'], format='%d/%m/%Y %H:%M')
        temporal_df['scat_id'] = temporal_df['scat_id'].astype(str)
        # Check for duplicates (optional, for debugging)
        duplicate_counts = temporal_df.groupby(['timestamp', 'scat_id']).size()
        duplicates = duplicate_counts[duplicate_counts > 1]
        if not duplicates.empty:
            print("Duplicate entries found:")
            print(duplicates)
        # Pivot with aggregation
        flow_matrix = temporal_df.pivot_table(
            index='timestamp',
            columns='scat_id',
            values='flow',
            aggfunc='sum'  # Adjust to 'mean' if needed
        )

        # Reindex columns to match scat_ids
        flow_matrix = flow_matrix.reindex(columns=scat_ids)

        # Handle missing values
        flow_matrix = flow_matrix.fillna(method='ffill').fillna(method='bfill')

        # Normalize flow values
        scaler = StandardScaler()
        flow_normalized = scaler.fit_transform(flow_matrix.values)  # Shape: [time, nodes]

        # Create data_tensor (add features if needed)
        data_tensor = flow_normalized[:, :, np.newaxis]  # Shape: [time, nodes, 1]
        # Get actual timestamps from temporal data
        timestamps = flow_matrix.index.to_pydatetime()
        return adj_matrix, data_tensor, scat_ids, coordinates, scaler, timestamps

    adj_matrix, data_tensor, scat_ids, coordinates, scaler, timestamps = preprocess_traffic_data(spatial_file, temporal_file)
    data_tensor = torch.tensor(data_tensor, dtype=torch.float32).to(device)

    # Load pre-trained STGNN model
    model = EnhancedSTGNN(
        input_dim=4,
        num_nodes=41,
        hidden_dim=64,  # Must match trained dimension
        output_dim=1,
        num_layers=3,
        dropout=0.1,
        window_size=24,
        horizon=12
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Parse and round current time
    current_time = datetime.strptime(current_time_str, '%d/%m/%Y %H:%M')
    rounded_time = round_to_nearest_15min(current_time)
    start_time = datetime.strptime(current_time_str, "%d/%m/%Y %H:%M")
    # Predict flows
    predicted_flows = predict_flows(model, data_tensor, start_time=start_time, actual_timestamps=timestamps)
    actual_flows = scaler.inverse_transform(predicted_flows)  # Back to vehicles/hour
    speed = 32 + np.sqrt(1024 - (256 / 375) * actual_flows)  # Speed in km/h
    print(predicted_flows)
    # Interpolate to 5-minute intervals (e.g., 36 intervals for 180 minutes)
    interpolated_flows = interpolate_flows(predicted_flows, num_intervals=3)

    # Update graph with initial travel times
    for u, v, data in G.edges(data=True):
        flow_u = interpolated_flows[0, scat_ids.index(u)]  # Flow at start time
        speed = calculate_speed(flow_u)
        travel_time = calculate_travel_time(data['distance'], speed)
        G[u][v]['weight'] = travel_time

    # Find 5 shortest paths
    paths = yen_k_shortest_paths(G, start_scat, dest_scat, k=5)

    # Output route details
    for i, path in enumerate(paths, 1):
        if not path:
            continue
        total_time = sum(G[path[j]][path[j+1]]['weight'] for j in range(len(path)-1))
        total_distance = sum(G[path[j]][path[j+1]]['distance'] for j in range(len(path)-1))
        segment_times = [G[path[j]][path[j+1]]['weight'] for j in range(len(path)-1)]
        print(f"Route {i}:")
        print(f"  Path: {' -> '.join(path)}")
        print(f"  Total Travel Time: {total_time:.2f} minutes")
        print(f"  Total Distance: {total_distance:.2f} km")
        print(f"  Segment Times: {', '.join([f'{t:.2f} min' for t in segment_times])}")

    # Identify crowded nodes over the next 30 minutes (6 intervals of 5 min)
    crowded_info = get_crowded_nodes(paths, interpolated_flows[:6], scat_ids)
    if crowded_info:
        print("\nNote: The following sites are predicted to be crowded in the next 30 minutes:")
        for node, times in crowded_info.items():
            print(f"  - SCATS {node} at {', '.join([f'{t} min' for t in times])}")

if __name__ == "__main__":
    # Example usage
    main(
        start_scat='1001',
        dest_scat='1040',
        current_time_str='12/2/2023 08:07',
        spatial_file='traffic_network2.csv',
        temporal_file='TrainingDataAdaptedOutput.csv',
        model_path='results/best_model.pt',
        output_dir='./results'
    )