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

# Assuming these are imported from claude_solution.py
# For standalone execution, I've included minimal definitions here
# Replace with actual imports if claude_solution.py is available
class EnhancedSTGNN(nn.Module):
    def __init__(self, input_dim=35, num_nodes=41, hidden_dim=64, output_dim=8, num_layers=3,
                 dropout=0.1, window_size=48, horizon=4, embedding_dim=16):
        super(EnhancedSTGNN, self).__init__()
        self.window_size = window_size
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.hour_embedding = nn.Linear(1, embedding_dim)
        self.day_embedding = nn.Linear(1, embedding_dim)
        self.weekend_embedding = nn.Linear(1, embedding_dim)
        self.num_directions = 8  # Fixed based on your input
        total_features = self.num_directions * 2 + embedding_dim * 3  # flow, imputed_flag, node_emb, hour_pe, day_pe, weekend_emb
        self.input_embedding = nn.Conv2d(total_features, hidden_dim, kernel_size=1)
        # Simplified for brevity; include full STGNN layers as in original
        self.fc = nn.Linear(hidden_dim * window_size, horizon * self.num_directions)

    def forward(self, x, static_adj=None, sampling_prob=1.0, mc_dropout_samples=10):
        batch_size = x.size(0)
        num_nodes = x.size(2)
        flow = x[..., :self.num_directions]
        imputed_flag = x[..., self.num_directions:self.num_directions*2]
        hour = x[..., self.num_directions*2:self.num_directions*2+1]
        weekend = x[..., self.num_directions*2+1:self.num_directions*2+2]
        day = x[..., self.num_directions*2+2:self.num_directions*2+3]
        node_emb = self.node_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, self.window_size, -1, -1)
        hour_emb = self.hour_embedding(hour)
        day_emb = self.day_embedding(day)
        weekend_emb = self.weekend_embedding(weekend)
        x = torch.cat([flow, imputed_flag, node_emb, hour_emb, day_emb, weekend_emb], dim=-1)
        x = x.permute(0, 3, 2, 1)
        x = self.input_embedding(x)
        x = x.reshape(batch_size, num_nodes, -1)
        out = self.fc(x).reshape(batch_size, num_nodes, self.horizon, self.num_directions)
        mean_out = out.permute(0, 2, 1, 3)
        std_out = torch.zeros_like(mean_out)
        return mean_out, std_out, []

def build_hybrid_adjacency_matrix(nodes, network_data=None, coordinates=None, threshold_distance=1.0, use_dynamic_threshold=True):
    num_nodes = len(nodes)
    A = np.zeros((num_nodes, num_nodes))
    if network_data is not None:
        A = network_data.copy()
    if coordinates is not None:
        coords_array = np.array([[coordinates[node]['lat'], coordinates[node]['lon']] for node in nodes])
        coords_rad = np.radians(coords_array)
        distances = haversine_distances(coords_rad) * 6371
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if distances[i, j] < threshold_distance and A[i, j] == 0:
                    A[i, j] = A[j, i] = 1
    np.fill_diagonal(A, 1)
    return A

def build_dynamic_adjacency_matrix(nodes, coordinates, flows, base_adj, flow_threshold=1000, max_distance=1.0):
    num_nodes = len(nodes)
    num_timesteps = flows.shape[0]
    dynamic_adj = np.zeros((num_timesteps, num_nodes, num_nodes))
    coords_array = np.array([[coordinates[node]['lat'], coordinates[node]['lon']] for node in nodes])
    coords_rad = np.radians(coords_array)
    distances = haversine_distances(coords_rad) * 6371
    for t in range(num_timesteps):
        adj_t = base_adj.copy()
        flow_t = flows[t]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if distances[i, j] < max_distance:
                    if flow_t[i] > flow_threshold or flow_t[j] > flow_threshold:
                        adj_t[i, j] = adj_t[j, i] = 0
                    elif adj_t[i, j] == 0:
                        adj_t[i, j] = adj_t[j, i] = 1
        np.fill_diagonal(adj_t, 1)
        dynamic_adj[t] = adj_t
    return dynamic_adj

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper functions (unchanged from original)
def haversine_distance(lat1, lon1, lat2, lon2):
    coords1 = np.radians([lat1, lon1])
    coords2 = np.radians([lat2, lon2])
    dist = haversine_distances([coords1, coords2]) * 6371
    return dist[0, 1]

def compute_speed(flow):
    if flow > 1500:
        return 5.0
    return 50 / (1 + math.exp(-0.002 * (flow - 500)))

def compute_travel_time(distance, flow):
    speed = compute_speed(flow)
    time_hours = distance / speed
    return time_hours * 60 + 0.5

def round_to_nearest_15min(dt):
    minute = dt.minute
    rounded_minute = round(minute / 15) * 15
    if rounded_minute == 60:
        dt = dt.replace(hour=dt.hour + 1, minute=0)
    else:
        dt = dt.replace(minute=rounded_minute)
    return dt.replace(second=0, microsecond=0)

def create_cyclical_encoding(values, max_value, embedding_dim):
    position = values.reshape(-1, 1)
    div_term = np.exp(np.arange(0, embedding_dim, 2) * (-np.log(10000.0) / embedding_dim))
    pe = np.zeros((len(values), embedding_dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

def prepare_input_data(temporal_df, start_time, window_size=48, num_nodes=41):
    start_time = round_to_nearest_15min(start_time)
    end_time = start_time - timedelta(minutes=15 * window_size)
    time_range = pd.date_range(end_time, start_time - timedelta(minutes=15), freq='15min')
    if len(time_range) != window_size:
        raise ValueError(f"Expected {window_size} time steps, got {len(time_range)}")

    df = temporal_df[temporal_df['timestamp'].isin(time_range)]
    if len(df) < window_size * num_nodes:
        raise ValueError("Incomplete historical data")

    # Temporal features with cyclical encodings
    hour_values = pd.to_datetime(time_range).hour / 23.0
    day_values = (pd.to_datetime(time_range).dayofweek + 1) / 7.0
    hour_pe = create_cyclical_encoding(hour_values, 24, embedding_dim=16)
    day_pe = create_cyclical_encoding(day_values, 7, embedding_dim=16)
    weekend = (pd.to_datetime(time_range).dayofweek >= 5).astype(float)[:, np.newaxis]

    # Pivot flow data
    directions = sorted(temporal_df['direction'].unique())
    num_directions = len(directions)  # Should be 8
    scat_ids = sorted(temporal_df['scat_id'].unique())
    full_multi_index = pd.MultiIndex.from_product([scat_ids, directions], names=['scat_id', 'direction'])
    flow_matrix = df.pivot(index='timestamp', columns=['scat_id', 'direction'], values='flow') \
                   .reindex(index=time_range, columns=full_multi_index, fill_value=0)
    imputed_mask = (flow_matrix == 0).astype(int)

    flow_values = np.stack([flow_matrix.xs(scat, level='scat_id', axis=1).values for scat in scat_ids], axis=1)
    imputed_values = np.stack([imputed_mask.xs(scat, level='scat_id', axis=1).values for scat in scat_ids], axis=1)
    flow_values_2d = flow_values.reshape(-1, num_directions)
    flow_normalized = scaler.transform(flow_values_2d)
    flow_normalized = flow_normalized.reshape(flow_values.shape)

    # Combine features
    time_features = np.concatenate([hour_values[:, np.newaxis], weekend, day_values[:, np.newaxis]], axis=-1)
    time_features = np.broadcast_to(time_features, (window_size, num_nodes, 3))
    input_tensor = np.concatenate([flow_normalized, imputed_values, time_features], axis=-1)
    return torch.FloatTensor(input_tensor).unsqueeze(0).to(device), num_directions

def predict_flows(model, input_tensor, static_adj, start_time, num_nodes, horizon=4, num_steps=12, mc_dropout_samples=10):
    predictions = []
    uncertainties = []
    current_input = input_tensor.clone()

    for step in range(0, num_steps, horizon):
        with torch.no_grad():
            pred, uncertainty, _ = model(current_input, static_adj, mc_dropout_samples=mc_dropout_samples)
        predictions.append(pred.cpu().numpy())
        uncertainties.append(uncertainty.cpu().numpy())

        if step + horizon < num_steps:
            horizon_steps = min(horizon, num_steps - step)
            pred_flow = pred[:, :horizon_steps, :, :]
            batch_size, time_steps, actual_nodes, num_directions = pred_flow.shape
            pred_flow_2d = pred_flow.reshape(-1, num_directions)
            pred_flow_2d = scaler.inverse_transform(pred_flow_2d)
            pred_flow_2d = scaler.transform(pred_flow_2d)
            pred_flow = pred_flow_2d.reshape(batch_size, time_steps, actual_nodes, num_directions)

            current_input = current_input[:, horizon:, :, :].clone()
            new_input = torch.zeros((1, horizon_steps, actual_nodes, current_input.shape[-1]), device=device)
            new_input[:, :, :, :num_directions] = torch.FloatTensor(pred_flow).to(device)
            new_input[:, :, :, num_directions:num_directions*2] = 0

            start_idx = step + horizon
            end_idx = start_idx + horizon
            time_range = pd.date_range(start_time + timedelta(minutes=15 * (start_idx + 1)),
                                       start_time + timedelta(minutes=15 * end_idx), freq='15min')
            hour_values = pd.to_datetime(time_range).hour / 23.0
            day_values = (pd.to_datetime(time_range).dayofweek + 1) / 7.0
            weekend = (pd.to_datetime(time_range).dayofweek >= 5).astype(float)[:, np.newaxis]
            time_features = np.concatenate([hour_values[:, np.newaxis], weekend, day_values[:, np.newaxis]], axis=-1)
            time_features = np.broadcast_to(time_features, (horizon_steps, num_nodes, 3))
            new_input[:, :, :, num_directions*2:] = torch.FloatTensor(time_features).to(device)
            current_input = torch.cat([current_input, new_input], dim=1)

    predictions = np.concatenate(predictions, axis=1)[:, :num_steps, :, :]
    uncertainties = np.concatenate(uncertainties, axis=1)[:, :num_steps, :, :]
    return predictions, uncertainties

def interpolate_flows(flows, uncertainties, num_steps=12, interval=15, target_interval=5):
    num_target_steps = int(num_steps * (interval / target_interval))
    time_points = np.arange(0, num_steps * interval, interval)
    target_time_points = np.arange(0, num_steps * interval, target_interval)
    flows_interpolated = np.zeros((flows.shape[0], num_target_steps, flows.shape[2], flows.shape[3]))
    uncertainties_interpolated = np.zeros_like(flows_interpolated)

    for batch in range(flows.shape[0]):
        for node in range(flows.shape[2]):
            for direction in range(flows.shape[3]):
                cs_flow = CubicSpline(time_points, flows[batch, :, node, direction])
                cs_uncertainty = CubicSpline(time_points, uncertainties[batch, :, node, direction])
                flows_interpolated[batch, :, node, direction] = cs_flow(target_time_points)
                uncertainties_interpolated[batch, :, node, direction] = cs_uncertainty(target_time_points)

    return flows_interpolated, uncertainties_interpolated

def find_shortest_paths(G, start, destination, k=5):
    try:
        paths = list(nx.shortest_simple_paths(G, start, destination, weight='weight', k=k))
        return paths
    except nx.NetworkXNoPath:
        return []

def route_finding_stgnn(spatial_file, temporal_file, model_path, start_scat, dest_scat, start_time_str, scaler_path=None):
    global scaler
    spatial_df = pd.read_csv(spatial_file)
    temporal_df = pd.read_csv(temporal_file)
    temporal_df['timestamp'] = pd.to_datetime(temporal_df['timestamp'], format='%d/%m/%Y %H:%M')

    scat_ids = sorted(spatial_df['SCATS Number'].unique())
    num_nodes = len(scat_ids)
    scat_to_idx = {scat: i for i, scat in enumerate(scat_ids)}
    coordinates = {row['SCATS Number']: {'lat': row['Latitude'], 'lon': row['Longitude']} for _, row in spatial_df.iterrows()}

    distances = {}
    for i, scat1 in enumerate(scat_ids):
        for j, scat2 in enumerate(scat_ids):
            if scat1 != scat2:
                dist = haversine_distance(coordinates[scat1]['lat'], coordinates[scat1]['lon'],
                                          coordinates[scat2]['lat'], coordinates[scat2]['lon'])
                distances[(scat1, scat2)] = dist

    if scaler_path and os.path.exists(scaler_path):
        scaler = torch.load(scaler_path)
    else:
        directions = temporal_df['direction'].unique()
        scat_ids_data = sorted(temporal_df['scat_id'].unique())
        full_multi_index = pd.MultiIndex.from_product([scat_ids_data, directions], names=['scat_id', 'direction'])
        flow_matrix = temporal_df.pivot(index='timestamp', columns=['scat_id', 'direction'], values='flow') \
                                .reindex(columns=full_multi_index, fill_value=0)
        flow_values = np.stack([flow_matrix.xs(scat, level='scat_id', axis=1).values for scat in scat_ids_data], axis=1)
        flow_values_2d = flow_values.reshape(-1, flow_values.shape[-1])
        scaler = StandardScaler()
        scaler.fit(flow_values_2d)
        torch.save(scaler, 'scaler.pt')

    num_directions = 8  # Based on your confirmation
    input_dim = num_directions * 2 + 3  # Adjusted for scalar temporal features; should be 35 with embeddings
    model = EnhancedSTGNN(input_dim=input_dim, num_nodes=num_nodes, output_dim=num_directions).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for _, row in spatial_df.iterrows():
        node = scat_to_idx[row['SCATS Number']]
        neighbors = [str(n) for n in str(row['Neighbours']).split(';') if n.strip()]
        for nbr in neighbors:
            if nbr in scat_to_idx:
                adj_matrix[node, scat_to_idx[nbr]] = 1
                adj_matrix[scat_to_idx[nbr], node] = 1
    static_adj = build_hybrid_adjacency_matrix(nodes=scat_ids, network_data=adj_matrix, coordinates=coordinates, threshold_distance=1.0)
    static_adj_tensor = torch.FloatTensor(static_adj).to(device)

    start_time = pd.to_datetime(start_time_str, format='%Y-%m-%d %H:%M:%S')
    input_tensor, num_directions = prepare_input_data(temporal_df, start_time, window_size=48, num_nodes=num_nodes)
    flows_pred, uncertainties_pred = predict_flows(model, input_tensor, static_adj_tensor, start_time, num_nodes)
    flows_pred = flows_pred[0]
    uncertainties_pred = uncertainties_pred[0]

    flows_pred_2d = flows_pred.reshape(-1, flows_pred.shape[-1])
    flows_pred_original = scaler.inverse_transform(flows_pred_2d).reshape(flows_pred.shape)
    uncertainties_pred_2d = uncertainties_pred.reshape(-1, uncertainties_pred.shape[-1])
    uncertainties_pred_original = scaler.scale_ * uncertainties_pred_2d.reshape(uncertainties_pred.shape)

    flows_interpolated, uncertainties_interpolated = interpolate_flows(flows_pred_original, uncertainties_pred_original)
    flows_interpolated = flows_interpolated[0]
    uncertainties_interpolated = uncertainties_interpolated[0]
    flows_interpolated_agg = flows_interpolated.sum(axis=-1)
    uncertainties_interpolated_agg = uncertainties_interpolated.sum(axis=-1)

    dynamic_adj = build_dynamic_adjacency_matrix(nodes=scat_ids, coordinates=coordinates, flows=flows_interpolated_agg,
                                                 base_adj=static_adj, flow_threshold=1000, max_distance=1.0)

    G = nx.DiGraph()
    for scat in scat_ids:
        G.add_node(scat)
    flows_at_start = flows_interpolated_agg[0]
    for i, scat1 in enumerate(scat_ids):
        for j, scat2 in enumerate(scat_ids):
            if dynamic_adj[0, i, j] == 1:
                flow = flows_at_start[i]
                dist = distances.get((scat1, scat2), float('inf'))
                if dist != float('inf'):
                    travel_time = compute_travel_time(dist, flow)
                    G.add_edge(scat1, scat2, weight=travel_time, distance=dist)

    paths = find_shortest_paths(G, start_scat, dest_scat, k=5)

    output = []
    for path_idx, path in enumerate(paths):
        path_details = {
            'path': path,
            'total_travel_time': 0.0,
            'total_distance': 0.0,
            'segment_times': [],
            'crowded_nodes': []
        }
        for i in range(len(path) - 1):
            scat1, scat2 = path[i], path[i + 1]
            idx1, idx2 = scat_to_idx[scat1], scat_to_idx[scat2]
            dist = distances.get((scat1, scat2), 0.0)
            flow = flows_at_start[idx1]
            travel_time = compute_travel_time(dist, flow)
            path_details['segment_times'].append(travel_time)
            path_details['total_travel_time'] += travel_time
            path_details['total_distance'] += dist
        for node in path:
            node_idx = scat_to_idx[node]
            for t in range(min(6, flows_interpolated_agg.shape[0])):
                flow = flows_interpolated_agg[t, node_idx]
                if flow > 1000:
                    time_minutes = t * 5
                    path_details['crowded_nodes'].append((node, time_minutes))
        output.append(path_details)

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
        start_time_str="2006-10-01 08:00:00",
        scaler_path=None  # Set to 'scaler.pt' if saved during training
    )