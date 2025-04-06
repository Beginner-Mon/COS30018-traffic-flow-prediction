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

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model classes (unchanged)
class SpatialAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SpatialAttention, self).__init__()
        self.query_conv = nn.Conv2d(hidden_dim, hidden_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(hidden_dim, hidden_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.static_graph_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, static_adj=None):
        batch_size, C, N, T = x.size()
        query = self.query_conv(x.permute(0, 1, 3, 2)).reshape(batch_size * T, -1, N).permute(0, 2, 1)
        key = self.key_conv(x.permute(0, 1, 3, 2)).reshape(batch_size * T, -1, N)
        value = self.value_conv(x.permute(0, 1, 3, 2)).reshape(batch_size * T, -1, N)
        attention = torch.bmm(query, key)
        attention = torch.nn.functional.softmax(attention / math.sqrt(C), dim=-1)
        if static_adj is not None:
            static_adj_expanded = static_adj.unsqueeze(0).repeat(batch_size * T, 1, 1)
            attention = attention.masked_fill(static_adj_expanded == 0, float('-inf'))
            attention = torch.nn.functional.softmax(attention, dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.reshape(batch_size, T, C, N).permute(0, 2, 3, 1)
        return out, attention.reshape(batch_size, T, N, N)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(TemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.neighbor_proj = nn.Linear(hidden_dim, hidden_dim)
        self.node_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def aggregate_neighbors(self, x, static_adj):
        batch_size, num_nodes, time_steps, channels = x.size()
        adj_norm = torch.nn.functional.normalize(static_adj, p=1, dim=-1)
        x_proj = self.neighbor_proj(x)
        neighbor_feat = torch.einsum('ij,bjtc->bitc', adj_norm, x_proj)
        combined_feat = torch.cat([x, neighbor_feat], dim=-1)
        node_feat = self.node_proj(combined_feat)
        return node_feat

    def forward(self, x, static_adj=None):
        batch_size, channels, num_nodes, time_steps = x.size()
        x = x.permute(0, 2, 3, 1)
        query = self.query(x).reshape(batch_size, num_nodes, time_steps, self.num_heads, self.head_dim)
        key = self.key(x).reshape(batch_size, num_nodes, time_steps, self.num_heads, self.head_dim)
        value = self.value(x).reshape(batch_size, num_nodes, time_steps, self.num_heads, self.head_dim)
        query = query.permute(0, 1, 3, 2, 4)
        key = key.permute(0, 1, 3, 4, 2)
        value = value.permute(0, 1, 3, 2, 4)
        scores = torch.matmul(query, key) / math.sqrt(self.head_dim)
        attention = torch.nn.functional.softmax(scores, dim=-1)
        out = torch.matmul(attention, value)
        out = out.permute(0, 1, 3, 2, 4).reshape(batch_size, num_nodes, time_steps, channels)
        out = self.fc_out(out)
        return out.permute(0, 3, 1, 2)

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = x + self.pe[:, :x.size(1), :, :x.size(3)]
        return x.permute(0, 3, 2, 1)

class STGNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(STGNNLayer, self).__init__()
        self.spatial_attention = SpatialAttention(hidden_dim)
        self.temporal_attention = TemporalAttention(hidden_dim)
        self.layer_norm1 = nn.LayerNorm([hidden_dim])
        self.layer_norm2 = nn.LayerNorm([hidden_dim])
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, static_adj=None):
        if static_adj is not None:
            static_adj = torch.nn.functional.normalize(static_adj.float(), p=1, dim=-1)
        residual = x
        x_spatial, attention = self.spatial_attention(x, static_adj)
        x = residual + self.dropout(x_spatial)
        x = x.permute(0, 3, 2, 1)
        x = self.layer_norm1(x)
        x = x.permute(0, 3, 2, 1)
        residual = x
        x_temporal = self.temporal_attention(x)
        x = residual + self.dropout(x_temporal)
        x = x.permute(0, 3, 2, 1)
        x = self.layer_norm2(x)
        residual = x
        x_ffn = self.ffn(x)
        x = residual + self.dropout(x_ffn)
        x = x.permute(0, 3, 2, 1)
        return x, attention

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x):
        batch, channels, nodes, time = x.size()
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch * nodes, time, channels)
        x = self.transformer.encoder(x)
        x = x.reshape(batch, nodes, time, channels)
        x = x.permute(0, 3, 1, 2)
        return x

class EnhancedSTGNN(nn.Module):
    def __init__(self, input_dim=49, num_nodes=41, hidden_dim=64, output_dim=8, num_layers=3,
                 dropout=0.1, window_size=48, horizon=4, embedding_dim=16):
        super().__init__()
        self.window_size = window_size
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        temporal_features = 2 * embedding_dim + 1  # hour_pe, day_pe, weekend
        self.num_directions = (input_dim - temporal_features) // 2
        if self.num_directions <= 0:
            raise ValueError(f"input_dim ({input_dim}) is too small for embedding_dim ({embedding_dim}).")
        total_features = (self.num_directions +  # flow
                         self.num_directions +  # imputed_flag
                         embedding_dim +  # node_emb
                         2 * embedding_dim +  # hour_pe, day_pe
                         embedding_dim)  # weekend_emb
        # Total: 8 + 8 + 16 + 32 + 16 = 80
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.weekend_embedding = nn.Linear(1, embedding_dim)
        self.input_embedding = nn.Conv2d(
            in_channels=total_features,
            out_channels=hidden_dim,
            kernel_size=1
        )
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.stgnn_layers = nn.ModuleList([
            STGNNLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.recurrent_layer = TransformerLayer(hidden_dim, num_heads=4, num_layers=2, dropout=dropout)
        self.pred_layer = nn.Linear(hidden_dim * window_size, horizon * self.num_directions)

    def forward(self, x, static_adj=None, sampling_prob=1.0, mc_dropout_samples=10):
        batch_size = x.size(0)
        num_nodes = x.size(2)
        num_directions = self.num_directions
        embedding_dim = self.embedding_dim
        flow = x[..., :num_directions]
        imputed_flag = x[..., num_directions:num_directions * 2]
        hour_pe = x[..., num_directions * 2:num_directions * 2 + embedding_dim]
        day_pe = x[..., num_directions * 2 + embedding_dim:num_directions * 2 + 2 * embedding_dim]
        weekend = x[..., num_directions * 2 + 2 * embedding_dim:num_directions * 2 + 2 * embedding_dim + 1]
        node_emb = self.node_embedding.unsqueeze(0).unsqueeze(0)
        node_emb = node_emb.expand(batch_size, self.window_size, -1, -1)
        weekend_emb = self.weekend_embedding(weekend)
        x_combined = torch.cat([
            flow, imputed_flag, node_emb, hour_pe, day_pe, weekend_emb
        ], dim=-1)
        x = x_combined.permute(0, 3, 2, 1)
        x = self.input_embedding(x)
        x = self.pos_encoding(x)
        attention_matrices = []
        for layer in self.stgnn_layers:
            x, attention = layer(x, static_adj)
            attention_matrices.append(attention)
        x = self.recurrent_layer(x)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, self.hidden_dim * self.window_size)
        if not self.training and mc_dropout_samples > 1:
            self.train(True)
            predictions = []
            for _ in range(mc_dropout_samples):
                out = self.pred_layer(x)
                out = out.reshape(batch_size, num_nodes, self.horizon, num_directions)
                out = out.permute(0, 2, 1, 3)
                predictions.append(out.unsqueeze(-1))
            predictions = torch.cat(predictions, dim=-1)
            mean_out = predictions.mean(dim=-1)
            std_out = predictions.std(dim=-1)
            self.train(False)
        else:
            out = self.pred_layer(x)
            out = out.reshape(batch_size, num_nodes, self.horizon, num_directions)
            mean_out = out.permute(0, 2, 1, 3)
            std_out = torch.zeros_like(mean_out)
        return mean_out, std_out, attention_matrices

# Helper functions
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

def prepare_input_data(temporal_df, start_time, window_size=48, num_nodes=41, embedding_dim=16):
    start_time = pd.to_datetime(start_time, format='%d-%m-%Y %H:%M')
    start_time = round_to_nearest_15min(start_time)
    timestamps = pd.date_range(start=start_time - timedelta(minutes=15 * (window_size - 1)),
                                end=start_time, freq='15min')
    scat_ids = sorted(temporal_df['scat_id'].unique())
    directions = sorted(temporal_df['direction'].unique())
    num_directions = len(directions)
    full_multi_index = pd.MultiIndex.from_product([scat_ids, directions], names=['scat_id', 'direction'])
    flow_matrix = (
        temporal_df.pivot(index='timestamp', columns=['scat_id', 'direction'], values='flow')
        .reindex(index=timestamps, columns=full_multi_index)
    )
    flow_matrix = flow_matrix.fillna(0)
    flow_values = np.stack([flow_matrix.xs(scat, level='scat_id', axis=1).values for scat in scat_ids], axis=1)
    imputed_values = np.zeros_like(flow_values)
    hour_values = pd.to_datetime(timestamps).hour.to_numpy()
    day_values = (pd.to_datetime(timestamps).dayofweek + 1).to_numpy()
    hour_pe = create_cyclical_encoding(hour_values / 23.0, 24, embedding_dim)
    day_pe = create_cyclical_encoding(day_values / 7.0, 7, embedding_dim)
    dayofweek_standard = pd.to_datetime(timestamps).dayofweek.to_numpy()
    dayofweek_custom = np.where(dayofweek_standard == 6, 1,
                                np.where(dayofweek_standard == 5, 7, dayofweek_standard + 2))
    weekend = ((dayofweek_custom == 1) | (dayofweek_custom == 7)).astype(int)[:, None]  # Shape: (48, 1)
    # Stack only cyclical features
    cyclical_features = np.stack([hour_pe, day_pe], axis=-1)  # Shape: (48, 2, 16)
    cyclical_features = np.broadcast_to(cyclical_features[:, None, :, :],
                                        (cyclical_features.shape[0], len(scat_ids), cyclical_features.shape[1],
                                         cyclical_features.shape[2]))  # Shape: (48, 41, 2, 16)
    cyclical_features = cyclical_features.reshape(cyclical_features.shape[0], cyclical_features.shape[1], -1)  # Shape: (48, 41, 32)
    # Broadcast weekend separately
    weekend = weekend[:, None, :]  # Shape: (48, 1, 1)
    weekend = np.broadcast_to(weekend, (weekend.shape[0], len(scat_ids), 1))  # Shape: (48, 41, 1)
    # Combine time features
    time_features = np.concatenate([cyclical_features, weekend], axis=-1)  # Shape: (48, 41, 33)
    input_tensor = np.concatenate([
        flow_values,  # (48, 41, 8)
        imputed_values,  # (48, 41, 8)
        time_features  # (48, 41, 33)
    ], axis=-1)  # Total: 8 + 8 + 33 = 49
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Shape: (1, 48, 41, 49)
    return input_tensor, num_directions

def predict_flows(model, input_tensor, static_adj, start_time, num_nodes, horizon=4, num_steps=12,
                  mc_dropout_samples=10, embedding_dim=16):
    # Convert numpy array to PyTorch tensor and move to device
    input_tensor = torch.FloatTensor(input_tensor).to(device)
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
            new_input[:, :, :, num_directions:num_directions * 2] = 0

            start_idx = step + horizon
            end_idx = start_idx + horizon
            time_range = pd.date_range(start_time + timedelta(minutes=15 * (start_idx + 1)),
                                       start_time + timedelta(minutes=15 * end_idx), freq='15min')

            hour_values = np.array([dt.hour for dt in time_range]) / 23.0
            day_values = (np.array([dt.dayofweek for dt in time_range]) + 1) / 7.0
            weekend = np.array([dt.dayofweek >= 5 for dt in time_range]).astype(float)[:, np.newaxis]

            hour_pe = create_cyclical_encoding(hour_values, 24, embedding_dim)
            day_pe = create_cyclical_encoding(day_values, 7, embedding_dim)

            hour_pe_expanded = np.zeros((horizon_steps, num_nodes, embedding_dim))
            day_pe_expanded = np.zeros((horizon_steps, num_nodes, embedding_dim))
            weekend_expanded = np.zeros((horizon_steps, num_nodes, 1))

            for i in range(num_nodes):
                hour_pe_expanded[:, i, :] = hour_pe
                day_pe_expanded[:, i, :] = day_pe
                weekend_expanded[:, i, :] = weekend

            time_features = np.concatenate([hour_pe_expanded, day_pe_expanded, weekend_expanded], axis=-1)
            new_input[:, :, :, num_directions * 2:] = torch.FloatTensor(time_features).to(device)
            current_input = torch.cat([current_input, new_input], dim=1)

    predictions = np.concatenate(predictions, axis=1)[:, :num_steps, :, :]
    uncertainties = np.concatenate(uncertainties, axis=1)[:, :num_steps, :, :]
    return predictions, uncertainties

def interpolate_flows(flows, uncertainties, num_steps=12, interval=15, target_interval=5):
    # Check the dimensionality of flows and handle accordingly
    if len(flows.shape) == 3:  # If it's 3D (time_steps, num_nodes, num_directions)
        flows = flows[np.newaxis, ...]  # Add batch dimension
        uncertainties = uncertainties[np.newaxis, ...]

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

    # If original input was 3D, return 3D output
    if len(flows.shape) == 3:
        return flows_interpolated[0], uncertainties_interpolated[0]
    return flows_interpolated, uncertainties_interpolated

def find_shortest_paths(G, start, destination, k=5):
    try:
        paths = list(nx.shortest_simple_paths(G, start, destination, weight='weight'))
        return paths[:k]
    except nx.NetworkXNoPath:
        print(f"No path exists between {start} and {destination} in the graph.")
        return []

def build_hybrid_adjacency_matrix(nodes, network_data=None, coordinates=None, threshold_distance=5.0,
                                  use_dynamic_threshold=True):
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
                    A[i, j] = A[j, i] = 1  # Ensure symmetry
    np.fill_diagonal(A, 1)
    print(f"Number of connections in static_adj (excluding diagonal): {np.sum(A) - num_nodes}")
    return A

def build_dynamic_adjacency_matrix(nodes, coordinates, flows, base_adj, flow_threshold=5000, max_distance=5.0):
    num_nodes = len(nodes)
    num_timesteps = flows.shape[0]
    print(f"Flows shape in build_dynamic_adjacency_matrix: {flows.shape}")
    print(f"Sample flows: {flows[0, :5]}")
    dynamic_adj = np.zeros((num_timesteps, num_nodes, num_nodes))
    coords_array = np.array([[coordinates[node]['lat'], coordinates[node]['lon']] for node in nodes])
    coords_rad = np.radians(coords_array)
    distances = haversine_distances(coords_rad) * 6371
    print(f"Max distance between nodes: {distances.max()} km")
    for t in range(num_timesteps):
        adj_t = base_adj.copy()
        flow_t = flows[t]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if distances[i, j] < max_distance:
                    if flow_t[i] > flow_threshold or flow_t[j] > flow_threshold:
                        adj_t[i, j] = adj_t[j, i] = 0  # Ensure symmetry
                    elif adj_t[i, j] == 0:
                        adj_t[i, j] = adj_t[j, i] = 1  # Ensure symmetry
        np.fill_diagonal(adj_t, 1)
        dynamic_adj[t] = adj_t
    return dynamic_adj


def find_shortest_paths_yen(G, source, target, k=5):
    """
    Implementation of Yen's algorithm for k-shortest paths
    """
    # Compute the shortest path
    if source not in G or target not in G:
        print(f"Source {source} or target {target} not in graph")
        return []

    try:
        shortest_path = nx.shortest_path(G, source, target, weight='weight')
        dist = sum(G[shortest_path[i]][shortest_path[i + 1]]['weight'] for i in range(len(shortest_path) - 1))
    except nx.NetworkXNoPath:
        print(f"No path exists between {source} and {target}")
        return []

    # Initialize variables
    A = [(dist, shortest_path)]  # List of shortest paths found so far
    B = []  # List of potential k-shortest paths

    # Main loop
    for i in range(1, k):
        # The path from the previous iteration
        prev_path = A[-1][1]

        # For each node in the previous path except the last one
        for j in range(len(prev_path) - 1):
            # Root is the path from source to the current node
            root = prev_path[:j + 1]
            spur_node = prev_path[j]

            # Edges to be removed
            edges_to_remove = []

            # Remove edges that are part of previous paths with the same root
            for path_dist, path in A:
                if len(path) > j and path[:j + 1] == root:
                    if j + 1 < len(path):
                        # Remove edge that leads to the next node in the path
                        u, v = path[j], path[j + 1]
                        if G.has_edge(u, v):
                            edges_to_remove.append((u, v, G[u][v]))

            # Remove nodes that are part of the root (except the spur node)
            nodes_to_remove = [node for node in root if node != spur_node]

            # Create a copy of the graph and remove edges and nodes
            G_copy = G.copy()
            for u, v, data in edges_to_remove:
                if G_copy.has_edge(u, v):
                    G_copy.remove_edge(u, v)
            for node in nodes_to_remove:
                if node in G_copy:
                    G_copy.remove_node(node)

            # Try to find a spur path
            try:
                spur_path = nx.shortest_path(G_copy, spur_node, target, weight='weight')
                # Calculate the weight of the spur path
                spur_dist = sum(G[spur_path[i]][spur_path[i + 1]]['weight'] for i in range(len(spur_path) - 1))

                # Complete path: root + spur - first node of spur (to avoid duplication)
                total_path = root + spur_path[1:]
                # Calculate total path weight
                total_dist = sum(G[total_path[i]][total_path[i + 1]]['weight'] for i in range(len(total_path) - 1))

                # Add to potential paths
                B.append((total_dist, total_path))
            except nx.NetworkXNoPath:
                continue

        # If no new paths found, break
        if not B:
            break

        # Sort potential paths by weight
        B.sort(key=lambda x: x[0])

        # Add the best potential path to A
        best_path = B.pop(0)
        A.append(best_path)

    # Return just the paths, not the distances
    return [path for _, path in A]

def compute_time_dependent_travel_times(paths, flows_interpolated, start_time, scat_to_idx, distances,
                                       crowd_window_minutes=30, flow_threshold=1000):
    """
    Compute time-dependent travel times for the given paths.
    crowd_window_minutes: Time window (in minutes) from start_time to check for crowded nodes.
    flow_threshold: Flow value above which a node is considered crowded.
    """
    results = []
    crowd_window_steps = crowd_window_minutes // 5  # Number of 5-minute steps to check

    for path in paths:
        start_time_minutes = 0
        total_travel_time = 0
        total_distance = 0
        segment_times = []
        node_arrival_times = {path[0]: start_time_minutes}

        for i in range(len(path) - 1):
            scat1, scat2 = path[i], path[i + 1]
            idx1, idx2 = scat_to_idx[scat1], scat_to_idx[scat2]
            dist = distances.get((scat1, scat2), 0.0)
            time_idx = min(int(node_arrival_times[scat1] // 5), flows_interpolated.shape[0] - 1)
            flow = flows_interpolated[time_idx, idx1]
            travel_time = compute_travel_time(dist, flow)
            segment_times.append((f"{scat1}-{scat2}", travel_time))
            total_travel_time += travel_time
            total_distance += dist
            node_arrival_times[scat2] = node_arrival_times[scat1] + travel_time

        crowded_nodes = []
        for node in path:
            node_idx = scat_to_idx[node]
            node_crowded = False
            for t in range(min(crowd_window_steps, flows_interpolated.shape[0])):  # First crowd_window_minutes
                flow = flows_interpolated[t, node_idx]
                time_minutes = t * 5
                time_str = (start_time + timedelta(minutes=time_minutes)).strftime("%H:%M")
                if flow > flow_threshold:
                    crowded_nodes.append((node, time_str))
                    node_crowded = True
                    break
            if not node_crowded:
                print(f"  Node {node} not crowded (flow ≤ {flow_threshold}) within {crowd_window_minutes} minutes")

        results.append({
            'path': path,
            'total_travel_time': total_travel_time,
            'total_distance': total_distance,
            'segment_times': segment_times,
            'crowded_nodes': crowded_nodes,
            'arrival_times': node_arrival_times
        })

    return results


def route_finding_stgnn(spatial_file, temporal_file, model_path, start_scat, dest_scat, start_time_str,
                        scaler_path=None, crowd_window_minutes=30, flow_threshold=1000):
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

    directions = sorted(temporal_df['direction'].unique())
    num_directions_model = len(directions)

    if scaler_path is None:
        scaler_path = "./results/scaler.pt"

    if os.path.exists(scaler_path):
        scaler = torch.load(scaler_path, map_location=torch.device('cpu'), weights_only=False)
        if scaler.mean_.shape[0] != num_directions_model:
            raise ValueError(f"Scaler expects {scaler.mean_.shape[0]} directions, but data has {num_directions_model}")
    else:
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please train the model first.")

    embedding_dim = 16
    input_dim = num_directions_model * 2 + embedding_dim * 2 + 1
    model = EnhancedSTGNN(
        input_dim=input_dim,
        num_nodes=num_nodes,
        hidden_dim=64,
        output_dim=num_directions_model,
        num_layers=3,
        dropout=0.1,
        window_size=48,
        horizon=4,
        embedding_dim=embedding_dim
    ).to(device)

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for _, row in spatial_df.iterrows():
        node = scat_to_idx[row['SCATS Number']]
        neighbors = [int(n) for n in str(row['Neighbours']).split(';') if n.strip()]
        for nbr in neighbors:
            if nbr in scat_to_idx:
                adj_matrix[node, scat_to_idx[nbr]] = 1
                adj_matrix[scat_to_idx[nbr], node] = 1
    static_adj = build_hybrid_adjacency_matrix(nodes=scat_ids, network_data=adj_matrix, coordinates=coordinates,
                                               threshold_distance=5.0)
    static_adj_tensor = torch.FloatTensor(static_adj).to(device)

    start_time = pd.to_datetime(start_time_str, format='%d-%m-%Y %H:%M')
    input_tensor, num_directions = prepare_input_data(temporal_df, start_time, window_size=48, num_nodes=num_nodes,
                                                      embedding_dim=embedding_dim)
    flows_pred, uncertainties_pred = predict_flows(model, input_tensor, static_adj_tensor, start_time, num_nodes,
                                                   embedding_dim=embedding_dim)

    flows_pred_2d = flows_pred.reshape(-1, flows_pred.shape[-1])
    flows_pred_original = scaler.inverse_transform(flows_pred_2d).reshape(flows_pred.shape)
    uncertainties_pred_2d = uncertainties_pred.reshape(-1, uncertainties_pred.shape[-1])
    scaler_scale = scaler.scale_.numpy() if isinstance(scaler.scale_, torch.Tensor) else scaler.scale_
    uncertainties_pred_original = scaler_scale * uncertainties_pred_2d.reshape(uncertainties_pred.shape)

    flows_interpolated, uncertainties_interpolated = interpolate_flows(flows_pred_original, uncertainties_pred_original)
    flows_interpolated = flows_interpolated[0]
    uncertainties_interpolated = uncertainties_interpolated[0]
    flows_interpolated_agg = flows_interpolated.sum(axis=-1)

    G = nx.Graph()
    for scat in scat_ids:
        G.add_node(scat)
    flows_at_start = flows_interpolated_agg[0]
    for _, row in spatial_df.iterrows():
        scat1 = row['SCATS Number']
        idx1 = scat_to_idx[scat1]
        neighbors = [int(n) for n in str(row['Neighbours']).split(';') if n.strip()]
        for scat2 in neighbors:
            if scat2 in scat_to_idx:
                dist = distances.get((scat1, scat2), float('inf'))
                if dist != float('inf'):
                    flow = flows_at_start[idx1]
                    travel_time = compute_travel_time(dist, flow)
                    G.add_edge(scat1, scat2, weight=travel_time, distance=dist)

    paths = find_shortest_paths_yen(G, start_scat, dest_scat, k=5)
    if not paths:
        return {'routes': []}  # Return empty routes list if no paths found

    output = compute_time_dependent_travel_times(
        paths, flows_interpolated_agg, start_time, scat_to_idx, distances,
        crowd_window_minutes=crowd_window_minutes, flow_threshold=flow_threshold
    )

    seen_paths = set()
    unique_output = []
    for details in output:
        path_tuple = tuple(details['path'])
        if path_tuple not in seen_paths:
            seen_paths.add(path_tuple)
            unique_output.append(details)
    unique_output.sort(key=lambda x: x['total_travel_time'])
    unique_output = unique_output[:5]

    # Format routes for JSON response
    routes = []
    for idx, details in enumerate(unique_output, 1):
        route = {
            'route_number': idx,
            'path': [str(node) for node in details['path']],
            'travel_time_minutes': round(details['total_travel_time'], 2),
            'total_distance_km': round(details['total_distance'], 2),
            'segment_times': {seg[0]: round(seg[1], 2) for seg in details['segment_times']},
            'crowded_nodes': [{'node': str(node), 'time': time} for node, time in details['crowded_nodes']],
            'arrival_times': {str(node): round(time, 2) for node, time in details['arrival_times'].items()}
        }
        routes.append(route)

    return {'routes': routes}  # Return JSON-compatible dictionary
if __name__ == "__main__":
    spatial_df = pd.read_csv("traffic_network2.csv")
    scat_ids = sorted(spatial_df['SCATS Number'].unique())

    while True:
        try:
            start_scat = int(input("Enter the starting SCATS ID: "))
            if start_scat not in scat_ids:
                print(f"Error: SCATS ID {start_scat} not found in the network. Please choose a valid SCATS ID.")
                print(f"Valid SCATS IDs: {scat_ids}")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid integer for the SCATS ID.")

    while True:
        try:
            dest_scat = int(input("Enter the destination SCATS ID: "))
            if dest_scat not in scat_ids:
                print(f"Error: SCATS ID {dest_scat} not found in the network. Please choose a valid SCATS ID.")
                print(f"Valid SCATS IDs: {scat_ids}")
                continue
            if dest_scat == start_scat:
                print("Error: Start and destination SCATS IDs must be different.")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid integer for the SCATS ID.")

    while True:
        start_time_str = input("Enter the start time (DD-MM-YYYY H:MM, e.g., 26-10-2006 8:00): ")
        try:
            start_time = pd.to_datetime(start_time_str, format='%d-%m-%Y %H:%M')
            break
        except ValueError:
            print("Error: Invalid date format. Please use DD-MM-YYYY H:MM (e.g., 26-10-2006 8:00).")

    # Call the function and get the result
    result = route_finding_stgnn(
        spatial_file="traffic_network2.csv",
        temporal_file="TrainingDataAdaptedOutput.csv",
        model_path="results/best_model.pt",
        start_scat=start_scat,
        dest_scat=dest_scat,
        start_time_str=start_time_str,
        scaler_path="results/scaler.pt",
        crowd_window_minutes=30,
        flow_threshold=1000
    )

    # Print results for terminal use
    routes = result['routes']
    if not routes:
        print(f"\nNo routes found between SCATS {start_scat} and SCATS {dest_scat}.")
    else:
        for route in routes:
            print(f"\nRoute {route['route_number']}:")
            print(f"  Path: {' -> '.join(route['path'])}")
            print(f"  Total Travel Time: {route['travel_time_minutes']} minutes")
            print(f"  Total Distance: {route['total_distance_km']} km")
            segment_times_str = ", ".join([f"{k}: {v} min" for k, v in route['segment_times'].items()])
            print(f"  Segment Times: {segment_times_str}")
            if route['crowded_nodes']:
                crowded_nodes_str = ", ".join([f"{n['node']} at {n['time']}" for n in route['crowded_nodes']])
                print(f"  Crowded Nodes: {crowded_nodes_str}")

        # Summarize crowded nodes
        if any(route['crowded_nodes'] for route in routes):
            print(f"\nNOTE: The following sites are predicted to be crowded in the next 30 minutes:")
            crowded = {}
            for route in routes:
                for node_info in route['crowded_nodes']:
                    node, time = node_info['node'], node_info['time']
                    if node not in crowded or time < crowded[node]:
                        crowded[node] = time
            for node, time in sorted(crowded.items(), key=lambda x: x[1]):
                print(f"- SCATS {node} at {time}")
        else:
            print(f"\nNo sites are predicted to be crowded in the next 30 minutes.")