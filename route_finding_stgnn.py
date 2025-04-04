import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from sklearn.metrics.pairwise import haversine_distances
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import StandardScaler
import math

# Import necessary classes from claude_solution.py
from claude_solution import EnhancedSTGNN, build_hybrid_adjacency_matrix, build_dynamic_adjacency_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute Haversine distance between two points in km."""
    coords1 = np.radians([lat1, lon1])
    coords2 = np.radians([lat2, lon2])
    dist = haversine_distances([coords1, coords2]) * 6371  # Earth radius in km
    return dist[0, 1]


def compute_speed(flow):
    """Compute speed based on flow using the given formula."""
    if flow > 1500:
        return 5.0  # km/h
    return 50 / (1 + math.exp(-0.002 * (flow - 500)))


def compute_travel_time(distance, flow):
    """Compute travel time in minutes."""
    speed = compute_speed(flow)  # km/h
    time_hours = distance / speed  # hours
    time_minutes = time_hours * 60  # minutes
    return time_minutes + 0.5  # Add 30 seconds (0.5 minutes) for intersection delay


def round_to_nearest_15min(dt):
    """Round a datetime to the nearest 15-minute interval."""
    minute = dt.minute
    rounded_minute = round(minute / 15) * 15
    if rounded_minute == 60:
        dt = dt.replace(hour=dt.hour + 1, minute=0)
    else:
        dt = dt.replace(minute=rounded_minute)
    return dt.replace(second=0, microsecond=0)


def prepare_input_data(temporal_df, start_time, window_size=48, num_nodes=41):
    """Prepare input data for the model."""
    start_time = round_to_nearest_15min(start_time)
    end_time = start_time - timedelta(minutes=15 * window_size)
    time_range = pd.date_range(end_time, start_time - timedelta(minutes=15), freq='15min')
    if len(time_range) != window_size:
        raise ValueError(f"Expected {window_size} time steps, but got {len(time_range)}")

    # Filter temporal data
    df = temporal_df[(temporal_df['timestamp'].isin(time_range)) & (temporal_df['direction'] == 'N')]
    if len(df) != window_size * num_nodes:
        raise ValueError("Incomplete historical data for the given time window")

    # Create temporal features
    temporal_features = pd.DataFrame({
        'timestamp': time_range,
        'hour': pd.to_datetime(time_range).hour / 23.0,
        'weekend': (pd.to_datetime(time_range).dayofweek >= 5).astype(int),
        'day': (pd.to_datetime(time_range).dayofweek + 1) / 7.0
    })

    # Pivot flow data
    flow_matrix = df.pivot(index='timestamp', columns='scat_id', values='flow')
    flow_matrix = flow_matrix.reindex(index=time_range, fill_value=0)
    imputed_mask = (flow_matrix == 0).astype(int)

    # Create input tensor
    flow_values = flow_matrix.values[:, :, np.newaxis]  # Shape: (window_size, nodes, 1)
    imputed_values = imputed_mask.values[:, :, np.newaxis]  # Shape: (window_size, nodes, 1)
    time_features = temporal_features[['hour', 'weekend', 'day']].values[:, np.newaxis, :]  # Shape: (window_size, 1, 3)
    time_features = np.broadcast_to(time_features, (window_size, num_nodes, 3))
    input_tensor = np.concatenate([flow_values, imputed_values, time_features],
                                  axis=-1)  # Shape: (window_size, nodes, 5)
    return torch.FloatTensor(input_tensor).unsqueeze(0).to(device), scaler.transform(flow_matrix.values)


def predict_flows(model, input_tensor, static_adj, start_time, num_nodes, horizon=4, num_steps=12,
                  mc_dropout_samples=10):
    """Predict flows for the next num_steps time steps."""
    predictions = []
    uncertainties = []
    current_input = input_tensor.clone()

    for step in range(0, num_steps, horizon):
        with torch.no_grad():
            pred, uncertainty, _ = model(current_input, static_adj, mc_dropout_samples=mc_dropout_samples)
        predictions.append(pred.cpu().numpy())  # Shape: (1, horizon, nodes)
        uncertainties.append(uncertainty.cpu().numpy())

        # Prepare next input by shifting and appending predictions
        if step + horizon < num_steps:
            pred_flow = pred[:, :min(horizon, num_steps - step), :].cpu().numpy()  # Shape: (1, horizon, nodes)
            pred_flow = pred_flow.reshape(-1, pred_flow.shape[-1])  # Shape: (horizon, nodes)
            pred_flow = scaler.inverse_transform(pred_flow)  # Inverse transform to original units
            pred_flow = scaler.transform(pred_flow)  # Transform back to normalized space
            pred_flow = pred_flow.reshape(1, -1, pred_flow.shape[-1], 1)  # Shape: (1, horizon, nodes, 1)

            # Update input tensor
            current_input = current_input[:, horizon:, :, :].clone()
            new_input = torch.zeros_like(current_input[:, :min(horizon, num_steps - step), :, :])
            new_input[:, :, :, 0:1] = torch.FloatTensor(pred_flow).to(device)  # Update flow
            new_input[:, :, :, 1:2] = 0  # Imputed flag for predicted values
            # Update temporal features
            start_idx = step + horizon
            end_idx = start_idx + horizon
            time_range = pd.date_range(
                start_time + timedelta(minutes=15 * (start_idx + 1)),
                start_time + timedelta(minutes=15 * (end_idx)),
                freq='15min'
            )
            temporal_features = pd.DataFrame({
                'timestamp': time_range,
                'hour': pd.to_datetime(time_range).hour / 23.0,
                'weekend': (pd.to_datetime(time_range).dayofweek >= 5).astype(int),
                'day': (pd.to_datetime(time_range).dayofweek + 1) / 7.0
            })
            time_features = temporal_features[['hour', 'weekend', 'day']].values[:, np.newaxis, :]
            time_features = np.broadcast_to(time_features, (min(horizon, num_steps - step), num_nodes, 3))
            new_input[:, :, :, 2:] = torch.FloatTensor(time_features).to(device)
            current_input = torch.cat([current_input, new_input], dim=1)

    predictions = np.concatenate(predictions, axis=1)[:, :num_steps, :]  # Shape: (1, num_steps, nodes)
    uncertainties = np.concatenate(uncertainties, axis=1)[:, :num_steps, :]  # Shape: (1, num_steps, nodes)
    return predictions, uncertainties


def interpolate_flows(flows, uncertainties, num_steps=12, interval=15, target_interval=5):
    """Interpolate flows from 15-minute intervals to 5-minute intervals."""
    num_target_steps = int(num_steps * (interval / target_interval))
    time_points = np.arange(0, num_steps * interval, interval)
    target_time_points = np.arange(0, num_steps * interval, target_interval)

    flows_interpolated = np.zeros((flows.shape[0], num_target_steps, flows.shape[2]))
    uncertainties_interpolated = np.zeros((uncertainties.shape[0], num_target_steps, uncertainties.shape[2]))

    for batch in range(flows.shape[0]):
        for node in range(flows.shape[2]):
            cs_flow = CubicSpline(time_points, flows[batch, :, node])
            cs_uncertainty = CubicSpline(time_points, uncertainties[batch, :, node])
            flows_interpolated[batch, :, node] = cs_flow(target_time_points)
            uncertainties_interpolated[batch, :, node] = cs_uncertainty(target_time_points)

    return flows_interpolated, uncertainties_interpolated


def find_shortest_paths(G, start, destination, k=5):
    """Find k shortest paths using Yen's algorithm."""
    try:
        paths = list(nx.shortest_simple_paths(G, start, destination, weight='weight', k=k))
        return paths
    except nx.NetworkXNoPath:
        return []


def route_finding_stgnn(spatial_file, temporal_file, model_path, start_scat, dest_scat, start_time_str,
                        scaler_path=None):
    """Find the 5 shortest paths with predicted flows and crowded nodes."""
    # Load data
    spatial_df = pd.read_csv(spatial_file)
    temporal_df = pd.read_csv(temporal_file)
    temporal_df['timestamp'] = pd.to_datetime(temporal_df['timestamp'], format='%d/%m/%Y %H:%M')

    scat_ids = sorted(spatial_df['SCATS Number'].unique())
    num_nodes = len(scat_ids)
    scat_to_idx = {scat: i for i, scat in enumerate(scat_ids)}
    coordinates = {
        row['SCATS Number']: {
            'lat': row['Latitude'],
            'lon': row['Longitude']
        } for _, row in spatial_df.iterrows()
    }

    # Compute distances between all pairs of nodes
    distances = {}
    for i, scat1 in enumerate(scat_ids):
        for j, scat2 in enumerate(scat_ids):
            if scat1 != scat2:
                dist = haversine_distance(
                    coordinates[scat1]['lat'], coordinates[scat1]['lon'],
                    coordinates[scat2]['lat'], coordinates[scat2]['lon']
                )
                distances[(scat1, scat2)] = dist

    # Load scaler (recompute if not provided)
    global scaler
    if scaler_path and os.path.exists(scaler_path):
        scaler = torch.load(scaler_path)
    else:
        flow_values = temporal_df[temporal_df['direction'] == 'N'].pivot(
            index='timestamp', columns='scat_id', values='flow'
        ).values
        scaler = StandardScaler()
        scaler.fit(flow_values)
        torch.save(scaler, 'scaler.pt')

    # Load model
    model = EnhancedSTGNN(
        input_dim=5,
        num_nodes=num_nodes,
        hidden_dim=64,
        output_dim=1,
        num_layers=3,
        dropout=0.1,
        window_size=48,
        horizon=4,
        embedding_dim=16
    ).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Build static adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for _, row in spatial_df.iterrows():
        node = scat_to_idx[row['SCATS Number']]
        neighbors = [str(n) for n in str(row['Neighbours']).split(';') if n.strip()]
        for nbr in neighbors:
            if nbr in scat_to_idx:
                adj_matrix[node, scat_to_idx[nbr]] = 1
                adj_matrix[scat_to_idx[nbr], node] = 1
    static_adj = build_hybrid_adjacency_matrix(
        nodes=scat_ids,
        network_data=adj_matrix,
        coordinates=coordinates,
        threshold_distance=1.0
    )
    static_adj_tensor = torch.FloatTensor(static_adj).to(device)

    # Parse start time
    start_time = pd.to_datetime(start_time_str, format='%Y-%m-%d %H:%M:%S')

    # Prepare input data and predict flows
    input_tensor, normalized_flows = prepare_input_data(temporal_df, start_time, window_size=48, num_nodes=num_nodes)
    flows_pred, uncertainties_pred = predict_flows(model, input_tensor, static_adj_tensor, start_time, num_nodes,
                                                   horizon=4, num_steps=12)
    flows_pred = flows_pred[0]  # Shape: (num_steps, nodes)
    uncertainties_pred = uncertainties_pred[0]

    # Inverse-transform flows and uncertainties
    flows_pred_2d = flows_pred.reshape(-1, flows_pred.shape[-1])
    flows_pred_original = scaler.inverse_transform(flows_pred_2d).reshape(flows_pred.shape)
    uncertainties_pred_2d = uncertainties_pred.reshape(-1, uncertainties_pred.shape[-1])
    uncertainties_pred_original = scaler.scale_ * uncertainties_pred_2d.reshape(uncertainties_pred.shape)

    # Interpolate flows to 5-minute intervals
    flows_interpolated, uncertainties_interpolated = interpolate_flows(
        flows_pred_original, uncertainties_pred_original, num_steps=12, interval=15, target_interval=5
    )
    flows_interpolated = flows_interpolated[0]  # Shape: (num_target_steps, nodes)
    uncertainties_interpolated = uncertainties_interpolated[0]

    # Build dynamic adjacency matrix
    dynamic_adj = build_dynamic_adjacency_matrix(
        nodes=scat_ids,
        coordinates=coordinates,
        flows=flows_interpolated,
        base_adj=static_adj,
        flow_threshold=1000,
        max_distance=1.0
    )

    # Find 5 shortest paths at the start time
    G = nx.DiGraph()
    for scat in scat_ids:
        G.add_node(scat)
    flows_at_start = flows_interpolated[0]  # Flows at the first 5-minute interval
    for i, scat1 in enumerate(scat_ids):
        for j, scat2 in enumerate(scat_ids):
            if dynamic_adj[0, i, j] == 1:
                flow = flows_at_start[i]
                dist = distances.get((scat1, scat2), float('inf'))
                if dist != float('inf'):
                    travel_time = compute_travel_time(dist, flow)
                    G.add_edge(scat1, scat2, weight=travel_time, distance=dist)

    paths = find_shortest_paths(G, start_scat, dest_scat, k=5)

    # Compute path details and identify crowded nodes
    output = []
    for path_idx, path in enumerate(paths):
        path_details = {
            'path': path,
            'total_travel_time': 0.0,
            'total_distance': 0.0,
            'segment_times': [],
            'crowded_nodes': []
        }

        # Compute total travel time and segment times
        for i in range(len(path) - 1):
            scat1, scat2 = path[i], path[i + 1]
            idx1, idx2 = scat_to_idx[scat1], scat_to_idx[scat2]
            dist = distances.get((scat1, scat2), 0.0)
            flow = flows_at_start[idx1]
            travel_time = compute_travel_time(dist, flow)
            path_details['segment_times'].append(travel_time)
            path_details['total_travel_time'] += travel_time
            path_details['total_distance'] += dist

        # Identify crowded nodes over the next 30 minutes (6 intervals at 5-minute steps)
        for node in path:
            node_idx = scat_to_idx[node]
            for t in range(min(6, flows_interpolated.shape[0])):
                flow = flows_interpolated[t, node_idx]
                if flow > 1000:
                    time_minutes = t * 5
                    path_details['crowded_nodes'].append((node, time_minutes))

        output.append(path_details)

    # Format output
    for path_idx, details in enumerate(output):
        print(f"\nPath {path_idx + 1}:")
        print(f"SCATS Sites: {details['path']}")
        print(f"Total Travel Time: {details['total_travel_time']:.2f} minutes")
        print(f"Total Distance: {details['total_distance']:.2f} km")
        print("Segment Travel Times (minutes):", [f"{t:.2f}" for t in details['segment_times']])
        if details['crowded_nodes']:
            print("Crowded Nodes (SCATS, Time in minutes):", details['crowded_nodes'])
        else:
            print("No crowded nodes predicted in the next 30 minutes.")


if __name__ == "__main__":
    route_finding_stgnn(
        spatial_file="traffic_network2.csv",
        temporal_file="TrainingDataAdaptedOutput.csv",
        model_path="results/best_model.pt",
        start_scat=970,
        dest_scat=2000,
        start_time_str="2006-10-01 08:00:00"
    )