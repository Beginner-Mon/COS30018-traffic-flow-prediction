import torch.nn as nn
import torch.nn.functional as F
import math
import random
from sklearn.impute import KNNImputer
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import haversine_distances
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)         # Seed for PyTorch (CPU)
np.random.seed(42)            # Seed for NumPy
random.seed(42)               # Seed for Python's random module
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)  # Seed for all GPUs
# how weights are initialized, how data is shuffled, or how random operations happen—changes every time you run it
def build_dynamic_adjacency_matrix(nodes, coordinates, flows, base_adj, flow_threshold=1000, max_distance=1.0): #use for route finding
    """
    Build a dynamic adjacency matrix based on predicted flows.
    
    Args:
        nodes: List of node IDs (SCATS numbers).
        coordinates: Dict of node coordinates {node: {'lat': lat, 'lon': lon}}.
        flows: Predicted flows, shape (timesteps, nodes).
        base_adj: Base static adjacency matrix, shape (num_nodes, num_nodes).
        flow_threshold: Flow threshold to adjust connectivity (default: 1000).
        max_distance: Maximum distance for adding new edges (default: 1.0 km).
    
    Returns:
        Dynamic adjacency matrix, shape (timesteps, num_nodes, num_nodes).
    """
    num_nodes = len(nodes)
    num_timesteps = flows.shape[0]
    dynamic_adj = np.zeros((num_timesteps, num_nodes, num_nodes))

    # Compute distances using Haversine formula
    coords_array = np.array([[coordinates[node]['lat'], coordinates[node]['lon']] for node in nodes])
    coords_rad = np.radians(coords_array)
    distances = haversine_distances(coords_rad) * 6371  # Distances in km

    for t in range(num_timesteps):
        adj_t = base_adj.copy()
        flow_t = flows[t]  # Flows at timestep t, shape (num_nodes,)

        # Adjust connectivity based on flows
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if distances[i, j] < max_distance:
                    # If flow at either node is high, reduce connectivity (simulate congestion)
                    if flow_t[i] > flow_threshold or flow_t[j] > flow_threshold:
                        adj_t[i, j] = adj_t[j, i] = 0  # Remove edge if congested
                    else:
                        # Add edge if within distance and not already connected
                        if adj_t[i, j] == 0:
                            adj_t[i, j] = adj_t[j, i] = 1

        # Add self-loops
        np.fill_diagonal(adj_t, 1)
        dynamic_adj[t] = adj_t

    return dynamic_adj

def build_hybrid_adjacency_matrix(nodes, network_data=None, coordinates=None, threshold_distance=1.0, use_dynamic_threshold=True):
    num_nodes = len(nodes)
    A = np.zeros((num_nodes, num_nodes))
    
    # Use existing network data if provided
    if network_data is not None:
        A = network_data.copy()
    
    # Calculate distances and apply threshold if coordinates are given
    if coordinates is not None:
        coords_array = np.array([[coordinates[node]['lat'], coordinates[node]['lon']] for node in nodes])
        coords_rad = np.radians(coords_array)
        distances = haversine_distances(coords_rad) * 6371  # Distances in km
        
        # Optional dynamic threshold calculation
        if use_dynamic_threshold and network_data is not None:
            connected_distances = []
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if A[i, j] == 1:
                        connected_distances.append(distances[i, j])
            if connected_distances:
                threshold_distance = np.percentile(connected_distances, 75)
                print(f"Dynamically calculated threshold: {threshold_distance:.2f} km")
                # Uncomment the following to visualize (optional)
                plt.hist(connected_distances, bins=20)
                plt.axvline(threshold_distance, color='r', linestyle='--', label=f'Threshold = {threshold_distance:.2f} km')
                plt.xlabel('Distance (km)')
                plt.ylabel('Frequency')
                plt.legend()
                plt.show()
        
        # Connect nodes within the threshold distance
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if distances[i, j] < threshold_distance and A[i, j] == 0:
                    A[i, j] = A[j, i] = 1  # Symmetric connection
    
    # Add self-loops
    np.fill_diagonal(A, 1)
    
    return A

class SpatialAttention(nn.Module):
    """Spatial attention module that learns the adjacency matrix dynamically"""
    def __init__(self, hidden_dim):
        super(SpatialAttention, self).__init__()
        self.query_conv = nn.Conv2d(hidden_dim, hidden_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(hidden_dim, hidden_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Initial static graph incorporation
        self.static_graph_weight = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, static_adj=None):
        # x shape: [batch, channels, nodes, time]
        batch_size, C, N, T = x.size()
        
        # Project to get query, key, value
        query = self.query_conv(x.permute(0, 1, 3, 2)).reshape(batch_size * T, -1, N).permute(0, 2, 1)
        key = self.key_conv(x.permute(0, 1, 3, 2)).reshape(batch_size * T, -1, N)
        value = self.value_conv(x.permute(0, 1, 3, 2)).reshape(batch_size * T, -1, N)
        
        # Tính điểm attention 
        attention = torch.bmm(query, key)  # [batch*T, nodes, nodes]
        
        # Chuẩn hóa attention scores trước khi kết hợp với static_adj
        attention = F.softmax(attention / math.sqrt(C), dim=-1)
        
        if static_adj is not None:
            static_adj_expanded = static_adj.unsqueeze(0).repeat(batch_size * T, 1, 1)
            attention = attention.masked_fill(static_adj_expanded == 0, float('-inf'))
            attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.reshape(batch_size, T, C, N).permute(0, 2, 3, 1)
        
        return out, attention.reshape(batch_size, T, N, N)


class TemporalAttention(nn.Module):
    """Temporal attention module that captures temporal dependencies with spatial context"""
    def __init__(self, hidden_dim, num_heads=4):
        super(TemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Các lớp xử lý neighbor aggregation
        self.neighbor_proj = nn.Linear(hidden_dim, hidden_dim)
        self.node_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Multi-head attention layers
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        
    def aggregate_neighbors(self, x, static_adj):
        """Tổng hợp features từ các node lân cận sử dụng static adjacency"""
        batch_size, num_nodes, time_steps, channels = x.size()    
        adj_norm = F.normalize(static_adj, p=1, dim=-1)
        
        # Project neighbor features
        x_proj = self.neighbor_proj(x)
        
        # Aggregate từ neighbors
        neighbor_feat = torch.einsum('ij,bjtc->bitc', adj_norm, x_proj)
        
        # Combine với node features
        combined_feat = torch.cat([x, neighbor_feat], dim=-1)
        node_feat = self.node_proj(combined_feat)
        
        return node_feat
        
    def forward(self, x, static_adj=None):
        """
        Forward pass của temporal attention với spatial context
        
        Args:
            x: Input tensor [batch, channels, nodes, time]
            static_adj: Static adjacency matrix [nodes, nodes]
            
        Returns:
            Output tensor cùng kích thước với input
        """
        # Reshape input: [batch, channels, nodes, time] -> [batch, nodes, time, channels]
        batch_size, channels, num_nodes, time_steps = x.size()
        x = x.permute(0, 2, 3, 1)
            
        # Reshape for multi-head attention
        query = self.query(x).reshape(batch_size, num_nodes, time_steps, self.num_heads, self.head_dim)
        key = self.key(x).reshape(batch_size, num_nodes, time_steps, self.num_heads, self.head_dim)
        value = self.value(x).reshape(batch_size, num_nodes, time_steps, self.num_heads, self.head_dim)
        
        # Transpose để có được shape phù hợp cho attention
        query = query.permute(0, 1, 3, 2, 4)  # [batch, nodes, heads, time, head_dim]
        key = key.permute(0, 1, 3, 4, 2)      # [batch, nodes, heads, head_dim, time]
        value = value.permute(0, 1, 3, 2, 4)  # [batch, nodes, heads, time, head_dim]
        
        # Tính attention scores
        scores = torch.matmul(query, key) / math.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, value)  # [batch, nodes, heads, time, head_dim]
        
        # Reshape và combine heads
        out = out.permute(0, 1, 3, 2, 4).reshape(batch_size, num_nodes, time_steps, channels)
        out = self.fc_out(out)
        
        # Return to original shape [batch, channels, nodes, time]
        return out.permute(0, 3, 1, 2)


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal information"""
    def __init__(self, hidden_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(2)  # [1, time, 1, hidden_dim]
        
        # Register buffer (not a parameter but should be saved)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [batch, channels, nodes, time]
        x = x.permute(0, 3, 2, 1)  # [batch, time, nodes, channels]
        x = x + self.pe[:, :x.size(1), :, :x.size(3)]
        return x.permute(0, 3, 2, 1)  # [batch, channels, nodes, time]


class STGNNLayer(nn.Module):
    """Combined spatial-temporal graph neural network layer"""
    def __init__(self, hidden_dim, dropout=0.1):
        super(STGNNLayer, self).__init__()
        self.spatial_attention = SpatialAttention(hidden_dim)
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # Residual connections and layer norms
        self.layer_norm1 = nn.LayerNorm([hidden_dim])
        self.layer_norm2 = nn.LayerNorm([hidden_dim])
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, x, static_adj=None):
        if static_adj is not None:
        # Normalize static adjacency
            static_adj = F.normalize(static_adj.float(), p=1, dim=-1)
            
        # Spatial attention with residual connection
        residual = x
        x_spatial, attention = self.spatial_attention(x, static_adj)
        x = residual + self.dropout(x_spatial)
        x = x.permute(0, 3, 2, 1)  # [batch, time, nodes, channels]
        x = self.layer_norm1(x)
        x = x.permute(0, 3, 2, 1)  # [batch, channels, nodes, time]
        
        # Temporal attention with residual connection
        residual = x
        x_temporal = self.temporal_attention(x)
        x = residual + self.dropout(x_temporal)
        x = x.permute(0, 3, 2, 1)  # [batch, time, nodes, channels]
        x = self.layer_norm2(x)
        
        # Feed-forward network with residual connection
        residual = x
        x_ffn = self.ffn(x)
        x = residual + self.dropout(x_ffn)
        x = x.permute(0, 3, 2, 1)  # [batch, channels, nodes, time]
        
        return x, attention

class TransformerLayer(nn.Module):
    """Transformer layer for capturing sequential patterns"""
    def __init__(self, hidden_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,  # Only using encoder part
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x):
        # x shape: [batch, channels, nodes, time]
        batch, channels, nodes, time = x.size()

        # Reshape for Transformer: [batch, nodes, time, channels] -> [batch * nodes, time, channels]
        x = x.permute(0, 2, 3, 1)  # [batch, nodes, time, channels]
        x = x.reshape(batch * nodes, time, channels)

        # Apply Transformer (using encoder only)
        x = self.transformer.encoder(x)  # [batch * nodes, time, channels]

        # Reshape back
        x = x.reshape(batch, nodes, time, channels)
        x = x.permute(0, 3, 1, 2)  # [batch, channels, nodes, time]

        return x
    
class EnhancedSTGNN(nn.Module):
    def __init__(self, input_dim=5, num_nodes=41, hidden_dim=64, output_dim=1, num_layers=3,
                 dropout=0.1, window_size=48, horizon=4, embedding_dim=16):
        super().__init__()
        self.window_size = window_size
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.hour_embedding = nn.Linear(1, embedding_dim)
        self.day_embedding = nn.Linear(1, embedding_dim)
        self.weekend_embedding = nn.Linear(1, embedding_dim)
        self.num_directions = input_dim - 4  # flow, imputed_flag, hour, weekend, day -> num_directions = 1
        # Include imputed_flag (1 channel) in total features
        total_features = self.num_directions + 1 + embedding_dim + 3 * embedding_dim  # 1 + 1 + 16 + 48 = 66
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
        self.pred_layer = nn.Linear(hidden_dim * window_size, horizon)
        
    def perturb_input(self, x, noise_level=0.1):
        noise = torch.randn_like(x) * noise_level
        return x + noise
        
    def forward(self, x, static_adj=None, sampling_prob=1.0, mc_dropout_samples=10):
        """
        Forward pass with Monte Carlo Dropout for uncertainty quantification.
        
        Args:
            x: Input tensor [batch, time, nodes, features]
            static_adj: Static adjacency matrix [nodes, nodes]
            sampling_prob: Sampling probability for training perturbation
            mc_dropout_samples: Number of MC Dropout samples for uncertainty estimation
        
        Returns:
            mean_out: Mean prediction [batch, horizon, nodes]
            std_out: Standard deviation of predictions [batch, horizon, nodes]
        """
        if self.training and sampling_prob < 1.0:
            x = self.perturb_input(x, noise_level=1.0 - sampling_prob)

        batch_size = x.size(0)
        num_nodes = x.size(2)

        # Split features
        flow = x[..., :self.num_directions]
        imputed_flag = x[..., self.num_directions:self.num_directions+1]  # New feature
        hour = x[..., self.num_directions+1:self.num_directions+2]
        weekend = x[..., self.num_directions+2:self.num_directions+3]
        day = x[..., self.num_directions+3:self.num_directions+4]

        # Create node embeddings
        node_emb = self.node_embedding.unsqueeze(0).unsqueeze(0)
        node_emb = node_emb.expand(batch_size, self.window_size, -1, -1)

        # Create time embeddings
        hour_emb = self.hour_embedding(hour)
        day_emb = self.day_embedding(day)
        weekend_emb = self.weekend_embedding(weekend)

        # Combine features and embeddings
        x = torch.cat([
            flow,
            imputed_flag,  # Include new feature
            node_emb,
            hour_emb,
            day_emb,
            weekend_emb
        ], dim=-1)

        # Apply layers
        x = x.permute(0, 3, 2, 1)
        x = self.input_embedding(x)
        x = self.pos_encoding(x)

        attention_matrices = []
        for layer in self.stgnn_layers:
            x, attention = layer(x, static_adj)
            attention_matrices.append(attention)

        x = self.recurrent_layer(x)

        # Reshape for prediction layer
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, self.hidden_dim * self.window_size)

        # Monte Carlo Dropout: Run multiple forward passes
        if not self.training and mc_dropout_samples > 1:
            self.train(True)  # Enable dropout during inference
            predictions = []
            for _ in range(mc_dropout_samples):
                out = self.pred_layer(x)
                out = out.reshape(batch_size, num_nodes, self.horizon)
                out = out.permute(0, 2, 1)
                predictions.append(out.unsqueeze(-1))  # [batch, horizon, nodes, 1]
            predictions = torch.cat(predictions, dim=-1)  # [batch, horizon, nodes, samples]
            mean_out = predictions.mean(dim=-1)  # [batch, horizon, nodes]
            std_out = predictions.std(dim=-1)  # [batch, horizon, nodes]
            self.train(False)  # Disable dropout after inference
        else:
            out = self.pred_layer(x)
            out = out.reshape(batch_size, num_nodes, self.horizon)
            mean_out = out.permute(0, 2, 1)
            std_out = torch.zeros_like(mean_out)  # No uncertainty during training

        return mean_out, std_out, attention_matrices

class MaskedLoss(nn.Module):
    """Base class for masked losses"""
    def __init__(self, null_val=0.0):
        super(MaskedLoss, self).__init__()
        self.null_val = null_val
        
    def _get_mask(self, target):
        if isinstance(self.null_val, float) and math.isnan(self.null_val):
            mask = ~torch.isnan(target)
        else:
            mask = (target != self.null_val)
        mask = mask.float()
        
        # Thêm kiểm tra mask
        if torch.sum(mask) == 0:
            print("Warning: Mask is all zeros!")
            return torch.ones_like(mask)
        
        # Normalize mask
        mask /= torch.mean(mask) + 1e-10
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        return mask
        
class MaskedMAE(MaskedLoss):
    """Mean Absolute Error with masking"""
    def forward(self, preds, target):
        mask = self._get_mask(target)
        loss = torch.abs(preds - target)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)
        
class MaskedMSE(MaskedLoss):
    """Mean Squared Error with masking"""
    def forward(self, preds, target):
        mask = self._get_mask(target)
        loss = torch.square(preds - target)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)
        
class MaskedHuber(MaskedLoss):
    """Huber Loss with masking"""
    def __init__(self, delta=1.0, null_val=0.0):
        super(MaskedHuber, self).__init__(null_val)
        self.delta = delta
        
    def forward(self, preds, target):
        mask = self._get_mask(target)
        diff = torch.abs(preds - target)
        
        # Huber loss
        huber_loss = torch.where(
            diff <= self.delta,
            0.5 * torch.square(diff),
            self.delta * (diff - 0.5 * self.delta)
        )
        
        loss = huber_loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

def train_stgnn(model, train_loader, val_loader, test_loader, 
                static_adj=None, optimizer=None, scheduler=None,
                num_epochs=50, patience=20, device='cuda',
                sampling_schedule='sigmoid', sampling_decay=0.98,
                clip_grad_norm=5.0, ckpt_path="best_model.pt",
                verbose=True, scaler=None):  # Added scaler parameter
    """
    Training framework for the STGNN model
    """
    # Initialize optimizer if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Initialize scheduler if not provided
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
    
    # Initialize loss functions
    criterion = MaskedHuber(delta=1.0)
    mse_loss = MaskedMSE()
    
    # Move adjacency matrix to device
    if static_adj is not None:
        static_adj = static_adj.to(device)
    
    # Training variables
    best_val_loss = float('inf')
    counter = 0
    
    # Store metrics
    train_losses = []
    val_losses = []
    val_maes = []
    val_rmses = []
    val_maes_original = []  # Store MAE in original units
    val_rmses_original = []  # Store RMSE in original units
    
    for epoch in range(num_epochs):
        sampling_prob = get_sampling_prob(epoch, num_epochs, schedule=sampling_schedule, k=sampling_decay)
        
        # Training
        model.train()
        epoch_loss = 0
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred, _, _ = model(x_batch, static_adj, sampling_prob=sampling_prob, mc_dropout_samples=1)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
            
            if verbose and (batch_idx % 10 == 0):
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.6f}, Sampling Prob: {sampling_prob:.4f}")
        
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        preds = []
        truths = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred, _, _ = model(x_batch, static_adj, mc_dropout_samples=10)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
                preds.append(y_pred.cpu().numpy())
                truths.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Compute metrics in normalized space
        preds = np.concatenate(preds, axis=0)
        truths = np.concatenate(truths, axis=0)
        val_mae = np.mean(np.abs(preds - truths))
        val_rmse = np.sqrt(np.mean((preds - truths) ** 2))
        val_maes.append(val_mae)
        val_rmses.append(val_rmse)
        
        # Inverse-transform predictions and ground truth to compute metrics in original units
        if scaler is not None:
            preds_2d = preds.reshape(-1, preds.shape[-1])
            truths_2d = truths.reshape(-1, truths.shape[-1])
            preds_original = scaler.inverse_transform(preds_2d).reshape(preds.shape)
            truths_original = scaler.inverse_transform(truths_2d).reshape(truths.shape)
            val_mae_original = np.mean(np.abs(preds_original - truths_original))
            val_rmse_original = np.sqrt(np.mean((preds_original - truths_original) ** 2))
            val_maes_original.append(val_mae_original)
            val_rmses_original.append(val_rmse_original)
        else:
            val_mae_original = val_mae
            val_rmse_original = val_rmse
            val_maes_original.append(val_mae_original)
            val_rmses_original.append(val_rmse_original)
        
        # Print epoch results including original units
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}, Val RMSE: {val_rmse:.6f}, "
                  f"Val MAE (original units): {val_mae_original:.2f} vehicles/15min, "
                  f"Val RMSE (original units): {val_rmse_original:.2f} vehicles/15min, "
                  f"Sampling Prob: {sampling_prob:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, ckpt_path)
            if verbose:
                print(f"Saved best model with validation loss: {best_val_loss:.6f}")
        else:
            counter += 1
        
        if counter >= patience:
            if verbose:
                print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Test the best model
    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    model.eval()
    
    test_loss = 0
    test_preds = []
    test_std = []
    test_truths = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred, y_std, _ = model(x_batch, static_adj, mc_dropout_samples=10)
            loss = criterion(y_pred, y_batch)
            test_loss += loss.item()
            test_preds.append(y_pred.cpu().numpy())
            test_std.append(y_std.cpu().numpy())
            test_truths.append(y_batch.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_preds = np.concatenate(test_preds, axis=0)
    test_std = np.concatenate(test_std, axis=0)
    test_truths = np.concatenate(test_truths, axis=0)
    test_mae = np.mean(np.abs(test_preds - test_truths))
    test_rmse = np.sqrt(np.mean((test_preds - test_truths) ** 2))
    
    # Compute test metrics in original units
    if scaler is not None:
        test_preds_2d = test_preds.reshape(-1, test_preds.shape[-1])
        test_truths_2d = test_truths.reshape(-1, test_truths.shape[-1])
        test_preds_original = scaler.inverse_transform(test_preds_2d).reshape(test_preds.shape)
        test_truths_original = scaler.inverse_transform(test_truths_2d).reshape(test_truths.shape)
        test_mae_original = np.mean(np.abs(test_preds_original - test_truths_original))
        test_rmse_original = np.sqrt(np.mean((test_preds_original - test_truths_original) ** 2))
    else:
        test_mae_original = test_mae
        test_rmse_original = test_rmse
    
    if verbose:
        print(f"Test Loss: {test_loss:.6f}, Test MAE: {test_mae:.6f}, Test RMSE: {test_rmse:.6f}, "
              f"Test MAE (original units): {test_mae_original:.2f} vehicles/15min, "
              f"Test RMSE (original units): {test_rmse_original:.2f} vehicles/15min")
    
    # Return results
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_maes': val_maes,
        'val_rmses': val_rmses,
        'val_maes_original': val_maes_original,  # Added
        'val_rmses_original': val_rmses_original,  # Added
        'test_loss': test_loss,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_mae_original': test_mae_original,  # Added
        'test_rmse_original': test_rmse_original,  # Added
        'predictions': test_preds,
        'uncertainty': test_std,
        'ground_truth': test_truths
    }

def get_sampling_prob(epoch, num_epochs, schedule='sigmoid', k=0.99):
    """
    Compute sampling probability based on epoch and decay schedule.
    
    Args:
        epoch: Current epoch
        num_epochs: Total number of epochs
        schedule: Decay type ('linear', 'exponential', 'sigmoid')
        k: Decay rate for exponential schedule
    Returns:
        Sampling probability (float between 0 and 1)
    """
    if schedule == 'linear':
        return 1.0 - epoch / num_epochs
    elif schedule == 'exponential':
        return k ** epoch
    elif schedule == 'sigmoid':
        return 1.0 / (1.0 + np.exp((epoch - num_epochs/2) / (num_epochs/10)))
    else:
        raise ValueError(f"Unknown schedule type: {schedule}")
    
def calculate_metrics_by_horizon(predictions, ground_truth):
    """
    Tính MAE và RMSE cho từng horizon
    
    Args:
        predictions: shape [batch, horizon, nodes]
        ground_truth: shape [batch, horizon, nodes]
    Returns:
        mae_by_horizon: List of MAE values for each horizon
        rmse_by_horizon: List of RMSE values for each horizon
    """
    mae_by_horizon = []
    rmse_by_horizon = []
    
    for h in range(predictions.shape[1]):  # For each horizon
        mae = np.mean(np.abs(predictions[:, h, :] - ground_truth[:, h, :]))
        rmse = np.sqrt(np.mean((predictions[:, h, :] - ground_truth[:, h, :]) ** 2))
        mae_by_horizon.append(mae)
        rmse_by_horizon.append(rmse)
    
    return mae_by_horizon, rmse_by_horizon

def run_stgnn_pipeline(data_file, network_file, output_dir='./results',
                      window_size=48, horizon=4, batch_size=32,
                      hidden_dim=64, num_layers=3, dropout=0.1,
                      num_epochs=50, learning_rate=0.001, patience=20):
    
    # 1. Preprocess Data
    adj_matrix, data_tensor, scat_ids, coordinates = preprocess_traffic_data(
        spatial_file=network_file,
        temporal_file=data_file,
        value_col="flow"
    )
    
    # 2. Normalize flow data
    scaler = StandardScaler()
    flow_values = data_tensor[:, :, 0]  # Take only flow value
    flow_normalized = scaler.fit_transform(flow_values)
    data_tensor[:, :, 0] = flow_normalized  # Update normalized flow data
    
    # 3. Create sliding windows
    X, y = [], []
    num_samples = len(data_tensor) - window_size - horizon + 1
    num_directions = data_tensor.shape[-1] - 3  # Assume last 3 features are temporal (hour, day, weekend)
    for i in range(num_samples):
        # Input window
        X.append(data_tensor[i:i+window_size])  # Shape: (window_size, nodes, features)
        # Target: only taking normalized flow data
        y.append(data_tensor[i+window_size:i+window_size+horizon, :, 0])  # Take first feature (flow) only
    
    X = np.array(X)  # Shape: (samples, window_size, nodes, features)
    y = np.array(y)  # Shape: (samples, horizon, nodes)
    
    # Data validation, for debugging purpose
    print("\nData structure after creating sliding windows:")
    print(f"X shape: {X.shape} (samples, window_size, nodes, features)")
    print(f"y shape: {y.shape} (samples, horizon, nodes)")
    print(f"X range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"y range: [{y.min():.4f}, {y.max():.4f}]")
    
    if np.isnan(X).any():
        print("Warning: NaN values detected in X")
        print("NaN locations in X:", np.where(np.isnan(X)))
    if np.isnan(y).any():
        print("Warning: NaN values detected in y")
        print("NaN locations in y:", np.where(np.isnan(y)))
    
    # 4. Split data
    print("\nSplitting data into train, validation and test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    
    # 5. Create DataLoaders
    class TrafficDataset(Dataset):
        def __init__(self, X, y):
            """
            Args:
                X: Input tensor (samples, window_size, nodes, features)
                y: Target tensor (samples, horizon, nodes)
            """
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
            
        def __len__(self):
            return len(self.X)
            
        def __getitem__(self, idx):
            # Không cần chuyển đổi shape vì đã đúng định dạng
            return self.X[idx], self.y[idx]
    
    train_dataset = TrafficDataset(X_train, y_train)
    val_dataset = TrafficDataset(X_val, y_val)
    test_dataset = TrafficDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # In thông tin về batch để kiểm tra
    print("\nBatch shapes:")
    x_sample, y_sample = next(iter(train_loader))
    print(f"Input batch: {x_sample.shape} (batch, window_size, nodes, features)")
    print(f"Target batch: {y_sample.shape} (batch, horizon, nodes)")
    
    # 6. Load or create adjacency matrix
    hybrid_adj = build_hybrid_adjacency_matrix(
        nodes=scat_ids,
        network_data=adj_matrix,
        coordinates=coordinates,
        threshold_distance=1.0,  # Distance in km, adjust as needed
    )
    static_adj = torch.FloatTensor(hybrid_adj).to(device)
    
    # 7. Initialize model
    print("Initializing STGNN model...")
    input_dim = data_tensor.shape[-1]  # directions + 3
    output_dim = 1  # Predicting flow only
    model = EnhancedSTGNN(
        input_dim=5,  # Updated to 5
        num_nodes=len(scat_ids),
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        window_size=window_size,
        horizon=horizon,
        embedding_dim=16
    ).to(device)
    
    # 8. Train model
    print("Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    results = train_stgnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        static_adj=static_adj,
        optimizer=optimizer,
        num_epochs=num_epochs,
        patience=patience,
        device=device,
        ckpt_path=os.path.join(output_dir, "best_model.pt"),
        scaler=scaler  # Pass the scaler to train_stgnn
    )
    
    # Sau khi training hoàn tất, tính và hiển thị detailed metrics
    print("\nDetailed Performance Metrics:")
    print("============================")
    
    # Tính metrics theo horizon
    mae_by_horizon, rmse_by_horizon = calculate_metrics_by_horizon(
        results['predictions'], 
        results['ground_truth']
    )
    
    # In kết quả
    print("\nMetrics by Prediction Horizon:")
    print("------------------------------")
    for h in range(horizon):
        print(f"Horizon {h+1:2d} ({(h+1)*15:2d} min): MAE = {mae_by_horizon[h]:.4f}, RMSE = {rmse_by_horizon[h]:.4f}")
    
    # Tính metrics tổng thể
    overall_mae = np.mean(mae_by_horizon)
    overall_rmse = np.mean(rmse_by_horizon)
    print("\nOverall Metrics:")
    print("---------------")
    print(f"Average MAE: {overall_mae:.4f}")
    print(f"Average RMSE: {overall_rmse:.4f}")
    
    # Visualize metrics by horizon
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, horizon + 1), mae_by_horizon, marker='o')
    plt.xlabel('Prediction Horizon (steps)')
    plt.ylabel('MAE')
    plt.title('MAE by Prediction Horizon')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, horizon + 1), rmse_by_horizon, marker='o', color='orange')
    plt.xlabel('Prediction Horizon (steps)')
    plt.ylabel('RMSE')
    plt.title('RMSE by Prediction Horizon')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'horizon_metrics.png'))
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Horizon': range(1, horizon + 1),
        'MAE': mae_by_horizon,
        'RMSE': rmse_by_horizon
    })
    metrics_df.to_csv(os.path.join(output_dir, 'horizon_metrics.csv'), index=False)
    
    # 9. Visualize results
    print("Creating result plots...")
    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(results['train_losses'], label='Train Loss')
    plt.plot(results['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot MAE and RMSE
    plt.subplot(1, 2, 2)
    plt.plot(results['val_maes'], label='Validation MAE')
    plt.plot(results['val_rmses'], label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Validation Metrics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    
    # 10. Save predictions
    # Compute variance of flows for each SCATS site to identify high-variance sites
    flow_values = data_tensor[:, :, 0]  # Shape: (timesteps, nodes)
    variances = np.var(flow_values, axis=0)  # Shape: (nodes,)
    variance_threshold = np.percentile(variances, 75)  # Top 25% variance
    high_variance_nodes = np.where(variances > variance_threshold)[0]

    # Apply Exponential Moving Average to high-variance nodes
    def apply_ema(data, alpha=0.1):
        """Apply Exponential Moving Average to a 1D array."""
        ema = np.copy(data)
        for t in range(1, len(data)):
            ema[t] = alpha * data[t] + (1 - alpha) * ema[t-1]
        return ema
    
    print("Saving prediction results...")
    # Reshape predictions and ground truth for inverse transform
    preds_3d = results['predictions']  # Shape: (samples, horizon, nodes)
    truth_3d = results['ground_truth']
    uncertainty_3d = results['uncertainty']  # Shape: (samples, horizon, nodes)

    # Apply EMA to high-variance nodes for both predictions and uncertainty
    for node_idx in high_variance_nodes:
        for sample_idx in range(preds_3d.shape[0]):
            preds_3d[sample_idx, :, node_idx] = apply_ema(preds_3d[sample_idx, :, node_idx])
            uncertainty_3d[sample_idx, :, node_idx] = apply_ema(uncertainty_3d[sample_idx, :, node_idx])

    # Reshape to 2D (samples*horizon, nodes) for inverse transform
    preds_2d = preds_3d.reshape(-1, preds_3d.shape[-1])
    truth_2d = truth_3d.reshape(-1, truth_3d.shape[-1])
    uncertainty_2d = uncertainty_3d.reshape(-1, uncertainty_3d.shape[-1])

    # Inverse transform using the scaler
    preds_original = scaler.inverse_transform(preds_2d).reshape(preds_3d.shape)
    truth_original = scaler.inverse_transform(truth_2d).reshape(truth_3d.shape)
    # Uncertainty is in the same normalized space as predictions, so apply the same transformation to std dev
    uncertainty_original = scaler.scale_ * uncertainty_2d.reshape(uncertainty_3d.shape)  # Scale std dev

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(os.path.join(output_dir, f'predictions_{timestamp}.npy'), preds_original)
    np.save(os.path.join(output_dir, f'ground_truth_{timestamp}.npy'), truth_original)
    np.save(os.path.join(output_dir, f'uncertainty_{timestamp}.npy'), uncertainty_original)


def preprocess_traffic_data(spatial_file, temporal_file, value_col="flow"):
    spatial_df = pd.read_csv(spatial_file)
    scat_ids = sorted(spatial_df['SCATS Number'].unique())
    num_nodes = len(scat_ids)
    scat_to_idx = {scat: i for i, scat in enumerate(scat_ids)}
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for _, row in spatial_df.iterrows():
        node = scat_to_idx[row['SCATS Number']]
        neighbors = [str(n) for n in str(row['Neighbours']).split(';') if n.strip()]
        for nbr in neighbors:
            if nbr in scat_to_idx:
                adj_matrix[node, scat_to_idx[nbr]] = 1
                adj_matrix[scat_to_idx[nbr], node] = 1
    coordinates = {
        row['SCATS Number']: {
            'lat': row['Latitude'], 
            'lon': row['Longitude']
        } for _, row in spatial_df.iterrows()
    }
    temporal_df = pd.read_csv(temporal_file)
    temporal_df['timestamp'] = pd.to_datetime(temporal_df['timestamp'], format='%d/%m/%Y %H:%M')
    temporal_df = temporal_df.groupby(['timestamp', 'scat_id', 'direction'])[value_col].sum().reset_index()
    unique_timestamps = sorted(temporal_df['timestamp'].unique())
    temporal_features = pd.DataFrame({
        'timestamp': unique_timestamps,
        'hour': pd.to_datetime(unique_timestamps).hour,
        'weekend': pd.to_datetime(unique_timestamps).dayofweek >= 5,
        'day': pd.to_datetime(unique_timestamps).dayofweek + 1
    }).set_index('timestamp')
    temporal_features['hour'] = temporal_features['hour'] / 23.0
    temporal_features['weekend'] = temporal_features['weekend'].astype(int)
    temporal_features['day'] = temporal_features['day'] / 7.0
    temporal_scats = temporal_df['scat_id'].unique()
    if set(temporal_scats) != set(scat_ids):
        missing = set(scat_ids) - set(temporal_scats)
        raise ValueError(f"Missing SCATS in temporal data: {missing}")
    directions = temporal_df['direction'].unique()
    scat_ids = sorted(temporal_df['scat_id'].unique())
    full_multi_index = pd.MultiIndex.from_product([scat_ids, directions], 
                                             names=['scat_id', 'direction'])
    flow_matrix = (
        temporal_df.pivot(
            index='timestamp',
            columns=['scat_id', 'direction'],
            values=value_col
        )
        .reindex(columns=full_multi_index, fill_value=0)
    )
    multi_columns = pd.MultiIndex.from_product([scat_ids, directions], names=['scat_id', 'direction'])
    flow_matrix = flow_matrix.reindex(columns=multi_columns)
    print("Applying KNN imputation for flow values...")
    imputed_mask = (flow_matrix == 0).astype(int)
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    flow_imputed = imputer.fit_transform(flow_matrix.values)
    if flow_imputed.shape[1] != flow_matrix.shape[1]:
        print(f"Warning: Number of columns in flow_imputed ({flow_imputed.shape[1]}) does not match flow_matrix ({flow_matrix.shape[1]}). Adjusting columns.")
        flow_imputed = flow_imputed[:, :flow_matrix.shape[1]]
    flow_df = pd.DataFrame(flow_imputed, index=flow_matrix.index, columns=flow_matrix.columns)
    imputed_df = pd.DataFrame(imputed_mask.values, index=flow_matrix.index, columns=flow_matrix.columns)
    print("Flow matrix shape after imputation:", flow_df.shape)
    print("Imputed mask shape:", imputed_df.shape)
    try:
        flow_values = np.stack(
            [flow_df.xs(scat, level='scat_id', axis=1).values for scat in scat_ids],
            axis=1
        )
        print("Flow values shape:", flow_values.shape)
        imputed_values = np.stack(
            [imputed_df.xs(scat, level='scat_id', axis=1).values for scat in scat_ids],
            axis=1
        )
        print("Imputed values shape:", imputed_values.shape)
    except Exception as e:
        print(f"Error during reshaping flow_values or imputed_values: {e}")
        raise
    missing_after = flow_df.isna().sum().sum()
    if missing_after > 0:
        print(f"Warning: {missing_after} missing values remain in flow data")
    else:
        print("All flow values have been successfully imputed")
    time_features = temporal_features.values[:, None, :]
    print("Time features shape:", time_features.shape)
    try:
        data_tensor = np.concatenate([
            flow_values[..., :1],
            imputed_values[..., :1],
            np.broadcast_to(time_features, 
                           (time_features.shape[0], flow_values.shape[1], time_features.shape[2]))
        ], axis=-1)
    except Exception as e:
        print(f"Error during data_tensor creation: {e}")
        raise
    print("\nFinal tensor shape:", data_tensor.shape)
    print("Features: [flow, imputed_flag, hour, weekend, day]")
    return adj_matrix, data_tensor, scat_ids, coordinates

if __name__ == "__main__":
    model = EnhancedSTGNN(input_dim=4, num_nodes=41, hidden_dim=64, output_dim=1)  
    model = model.to(device)
    # Specify your file paths here
    run_stgnn_pipeline(
        data_file="TrainingDataAdaptedOutput.csv",  # Temporal data
        network_file="traffic_network2.csv",        # Spatial data
        output_dir="./results",                      # Custom output directory
        window_size=48,       # 24 time steps as historical context
        horizon=4,           # Predict next 12 time steps
        batch_size=32, 
        hidden_dim=64,
        num_layers=3, 
        dropout=0.1,
        num_epochs=50,
        learning_rate=0.001,
        patience=20
    )

    