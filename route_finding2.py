import networkx as nx
import numpy as np
import tensorflow as tf
from keras.api.layers import Dense, LSTM, GRU, Embedding, Concatenate, Input, Reshape
from keras.api.models import Model, load_model
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.api.callbacks import EarlyStopping
import tkinter as tk
from tkinter import ttk
from threading import Thread
import queue
import time
import os

def create_road_network(csv_file='traffic_network2.csv'):
    """
    Create a road network from traffic_network2.csv
    
    Args:
        csv_file: Path to the CSV file containing SCATS site data with neighbors
        
    Returns:
        A NetworkX DiGraph representing the road network
    """
    G = nx.DiGraph()
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Add nodes
    for _, row in df.iterrows():
        scats_id = int(row['SCATS Number'])
        G.add_node(
            scats_id,
            name=row['Site Description'],
            type=row['Site Type'],
            latitude=float(row['Latitude']),
            longitude=float(row['Longitude'])
        )
    
    # Add edges based on the processed neighbors from traffic_network2.csv
    for _, row in df.iterrows():
        source_id = int(row['SCATS Number'])
        if pd.notna(row['Neighbours']) and row['Neighbours']:
            neighbours = row['Neighbours'].split(';')
            for neighbour in neighbours:
                try:
                    target_id = int(neighbour.strip())
                    if target_id in G.nodes:
                        # Calculate distance using geographic coordinates
                        source_lat = G.nodes[source_id]['latitude']
                        source_long = G.nodes[source_id]['longitude']
                        target_lat = G.nodes[target_id]['latitude']
                        target_long = G.nodes[target_id]['longitude']
                        
                        # Calculate geodesic distance using simplified approximation
                        distance = np.sqrt((source_lat - target_lat)**2 + 
                                        (source_long - target_long)**2) * 111  # 1 degree ≈ 111 km
                        
                        # Default edge attributes: 60 km/h converted to km/min
                        speed_limit = 60 / 60  # 1 km/min (60 km/h to km/min)
                        controlled = 'INT' in G.nodes[source_id]['type']
                        
                        G.add_edge(
                            source_id,
                            target_id,
                            distance=distance,
                            speed_limit=speed_limit,
                            controlled=controlled
                        )
                except ValueError:
                    print(f"Warning: Invalid neighbor ID {neighbour} for node {source_id}")
    
    return G

def calculate_travel_time(G, site_A, site_B, traffic_flow):
    """
    Calculate travel time between two connected SCATS sites
    
    Args:
        G: Road network graph
        site_A, site_B: SCATS site IDs
        traffic_flow: Current/predicted traffic flow at site A
        
    Returns:
        Travel time in minutes
    """
    try:
        edge_data = G.get_edge_data(site_A, site_B)
        if edge_data is None:
            return float('inf')  # Return infinity if no direct edge exists
        
        distance = edge_data['distance']  # in km
        speed_limit = edge_data['speed_limit']  # km/min (1 km/min = 60 km/h)
        
        # Traffic delay: 30 seconds (0.5 minutes) base delay for controlled intersections
        base_delay = 0.5 if edge_data['controlled'] else 0  # 30 seconds in minutes
        traffic_factor = 1.0 + (traffic_flow / 1000.0)  # Scale traffic impact
        
        travel_time = (distance / speed_limit) + (base_delay * traffic_factor)
        return travel_time
    except KeyError:
        return float('inf')

def load_traffic_data(file='TrainingDataAdaptedOutput.csv', lag=12):
    """
    Load traffic flow data from CSV file with the provided format.
    
    Args:
        file: Path to the CSV file
        lag: Number of time steps to use as input sequence
    
    Returns:
        DataFrame with site_id, timestamp, traffic_flow, and sequence
    """
    # Read the Excel file
    df = pd.read_csv(file)
    
    # Ensure the DataFrame has the expected columns
    df.columns = ['DateTime', 'Traffic_Flow', 'Site_Type', 'Observed', 'SCATS_Number']
    
    # Convert DateTime to pandas Timestamp
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Group by SCATS_Number to handle multiple sites
    data = []
    unique_sites = df['SCATS_Number'].unique()
    
    for site_id in unique_sites:
        site_data = df[df['SCATS_Number'] == site_id].sort_values('DateTime')
        flow = site_data['Traffic_Flow'].values
        
        # Create sequences for each site
        for i in range(len(flow) - lag):
            data.append({
                'site_id': int(site_id),  # Ensure integer for consistency
                'timestamp': site_data['DateTime'].iloc[i + lag],
                'traffic_flow': flow[i + lag],
                'sequence': flow[i:i + lag].tolist()  # Store as list for flexibility
            })
    
    return pd.DataFrame(data)

def prepare_data(data, seq_length=12):
    X_seq, X_site, y = [], [], []
    for _, row in data.iterrows():
        X_seq.append(np.array(row['sequence']))
        X_site.append(row['site_id'])
        y.append(row['traffic_flow'])
    
    X_seq = np.array(X_seq)
    X_site = np.array(X_site)
    y = np.array(y)
    
    # Reshape X_seq to include the feature dimension (1 for traffic flow)
    X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], 1)
    
    # Split into train/test sets (80/20)
    train_size = int(len(X_seq) * 0.8)
    X_seq_train, X_seq_test = X_seq[:train_size], X_seq[train_size:]
    X_site_train, X_site_test = X_site[:train_size], X_site[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return (X_seq_train, X_site_train, y_train, X_seq_test, X_site_test, y_test)

def build_model(model_type, seq_length=12, num_sites=41):
    time_series_input = Input(shape=(seq_length, 1), name='time_series')
    site_id_input = Input(shape=(1,), name='site_id')
    
    # Site embedding for all 41 sites
    site_embedding = tf.keras.layers.Embedding(num_sites+1, 8)(site_id_input)
    site_embedding = Reshape((8,))(site_embedding)
    
    if model_type == "lstm":
        x = LSTM(64, return_sequences=True)(time_series_input)
        x = LSTM(64)(x)
    elif model_type == "gru":
        x = GRU(64, return_sequences=True)(time_series_input)
        x = GRU(64)(x)
    else:  # saes (simplified as a deep feed-forward network)
        x = Dense(400, activation='relu')(time_series_input)
        x = Dense(400, activation='relu')(x)
        x = Dense(400, activation='relu')(x)
        x = Reshape((400,))(x)  # Flatten for concatenation
    
    combined = Concatenate()([x, site_embedding])
    output = Dense(1, activation='linear')(combined)
    
    model = Model(inputs=[time_series_input, site_id_input], outputs=output)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mape'])
    return model

def train_model(model, X_seq_train, X_site_train, y_train, name):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    hist = model.fit(
        [X_seq_train, X_site_train],
        y_train,
        batch_size=256,
        epochs=470,
        validation_split=0.05,
        callbacks=[early_stopping],
        verbose=1
    )
    model.save(f"model/{name}.keras")
    pd.DataFrame(hist.history).to_csv(f"model/{name}_loss.csv", index=False)
    return model

def evaluate_model(model, X_test, y_test, site_ids_test):
    y_pred = model.predict([X_test, site_ids_test.reshape(-1, 1)], verbose=0)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    site_metrics = {}
    unique_site_ids = np.unique(site_ids_test)
    for site_id in unique_site_ids:
        site_mask = site_ids_test == site_id
        if np.sum(site_mask) > 0:
            site_mae = mean_absolute_error(y_test[site_mask], y_pred[site_mask])
            site_rmse = np.sqrt(mean_squared_error(y_test[site_mask], y_pred[site_mask]))
            site_metrics[site_id] = {'MAE': site_mae, 'RMSE': site_rmse}
    
    return {
        'Overall MAE': mae,
        'Overall RMSE': rmse,
        'Site-specific Metrics': site_metrics
    }

def prepare_current_data(site_ids, file='TrainingDataAdaptedOutput.csv'):
    """
    Prepare the most recent data for prediction using actual data from TrainingDataAdaptedOutput.csv
    """
    df = pd.read_csv(file)
    df.columns = ['DateTime', 'Traffic_Flow', 'Site_Type', 'Speed_Limit', 'SCATS_Number']
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    current_data = []
    for site_id in site_ids:
        site_data = df[df['SCATS_Number'] == site_id].sort_values('DateTime')
        if not site_data.empty:
            # Get the last 24 hours (96 measurements at 15-min intervals)
            last_24h = site_data.tail(96)  # 24 hours * 4 measurements/hour = 96
            for _, row in last_24h.iterrows():
                current_data.append({
                    'site_id': int(site_id),
                    'timestamp': row['DateTime'],
                    'traffic_flow': row['Traffic_Flow']
                })
    
    return pd.DataFrame(current_data)

def update_graph_weights(G, predictions):
    for u, v in G.edges():
        traffic_flow = predictions.get(u, 0)
        travel_time = calculate_travel_time(G, u, v, traffic_flow)
        G[u][v]['weight'] = travel_time

def yen_k_shortest_paths(G, source, target, k=5, timeout=10):
    """
    Find up to 5 shortest paths using Yen's algorithm with timeout
    """
    try:
        start_time = time.time()
        shortest_path = nx.shortest_path(G, source, target, weight='weight')
        shortest_time = sum(G[shortest_path[i]][shortest_path[i+1]]['weight'] 
                          for i in range(len(shortest_path)-1))
        
        paths = [(shortest_path, shortest_time)]
        potential_paths = []
        G_original = G.copy()
        
        for _ in range(k-1):
            if time.time() - start_time > timeout:
                print("Route finding timed out")
                break
                
            last_path = paths[-1][0]
            for i in range(len(last_path)-1):
                spur_node = last_path[i]
                root_path = last_path[:i+1]
                G_temp = G_original.copy()
                for p, _ in paths:
                    if len(p) > i and p[:i+1] == root_path:
                        if G_temp.has_edge(p[i], p[i+1]):
                            G_temp.remove_edge(p[i], p[i+1])
                
                try:
                    spur_path = nx.shortest_path(G_temp, spur_node, target, weight='weight')
                    total_path = root_path[:-1] + spur_path
                    total_time = sum(G_original[total_path[j]][total_path[j+1]]['weight']
                                   for j in range(len(total_path)-1))
                    if total_path not in [p for p, _ in paths] and (total_path, total_time) not in potential_paths:
                        potential_paths.append((total_path, total_time))
                except nx.NetworkXNoPath:
                    continue
            
            if not potential_paths:
                break
            potential_paths.sort(key=lambda x: x[1])
            paths.append(potential_paths.pop(0))
        
        return paths[:k]
    except nx.NetworkXNoPath:
        print(f"No path found between {source} and {target}")
        return []
    except Exception as e:
        print(f"Error in path finding: {str(e)}")
        return []

class RouteFinderUI:
    def __init__(self, G, site_ids):
        self.G = G
        self.site_ids = site_ids
        self.root = tk.Tk()
        self.root.title("Route Finder")
        self.result_queue = queue.Queue()
        self.model = None
        
        tk.Label(self.root, text="Model Type:").grid(row=0, column=0, padx=5, pady=5)
        self.model_combo = ttk.Combobox(self.root, values=["lstm", "gru", "saes"])
        self.model_combo.set("lstm")
        self.model_combo.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(self.root, text="Start Node:").grid(row=1, column=0, padx=5, pady=5)
        self.start_combo = ttk.Combobox(self.root, values=site_ids)
        self.start_combo.grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(self.root, text="End Node:").grid(row=2, column=0, padx=5, pady=5)
        self.end_combo = ttk.Combobox(self.root, values=site_ids)
        self.end_combo.grid(row=2, column=1, padx=5, pady=5)
        
        self.train_button = tk.Button(self.root, text="Train/Load Model", command=self.train_or_load_model)
        self.train_button.grid(row=3, column=0, pady=10)
        
        self.find_button = tk.Button(self.root, text="Find Routes", command=self.find_routes, state='disabled')
        self.find_button.grid(row=3, column=1, pady=10)
        
        self.result_text = tk.Text(self.root, height=20, width=80)
        self.result_text.grid(row=4, column=0, columnspan=2, padx=5, pady=5)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def train_or_load_model(self):
        self.train_button.config(state='disabled')
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Training/Loading model...\n")
        thread = Thread(target=self.train_or_load_model_thread)
        thread.start()

    def train_or_load_model_thread(self):
        model_type = self.model_combo.get()
        model_path = f"model/{model_type}.keras"
        
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                result = f"Loaded pre-trained {model_type} model!\n"
            except Exception as e:
                self.result_text.insert(tk.END, f"Error loading model: {str(e)}\n")
                self.train_model_thread()
                return
        else:
            self.train_model_thread()
            return
        
        self.result_queue.put(result)
        self.find_button.config(state='normal')

    def train_model_thread(self):
        data = load_traffic_data()
        X_seq_train, X_site_train, y_train, _, _, _ = prepare_data(data)
        X_seq_train = X_seq_train  # Already reshaped in prepare_data
        
        model_type = self.model_combo.get()
        if not os.path.exists('model'):
            os.makedirs('model')
        
        self.model = build_model(model_type, num_sites=len(self.site_ids))
        self.model = train_model(self.model, X_seq_train, X_site_train, y_train, model_type)
        
        result = f"Model ({model_type}) trained!\n"
        self.result_queue.put(result)

    def find_routes(self):
        self.find_button.config(state='disabled')
        self.result_text.delete(1.0, tk.END)
        start = int(self.start_combo.get())
        end = int(self.end_combo.get())
        thread = Thread(target=self.calculate_routes, args=(start, end))
        thread.start()
        self.root.after(100, self.check_queue)

    def calculate_routes(self, start, end):
        data = load_traffic_data()
        X_seq, X_site, _ = prepare_data(data)
        
        predictions = {}
        pred = self.model.predict([X_seq, X_site], verbose=0)
        for i, site_id in enumerate(X_site):
            predictions[site_id] = pred[i][0]
        
        update_graph_weights(self.G, predictions)
        routes = yen_k_shortest_paths(self.G, start, end, k=5)
        
        result = f"Finding up to 5 routes from {start} to {end}:\n\n"
        if routes:
            for i, (path, time) in enumerate(routes):
                result += f"Route {i+1}: {path}\n"
                result += f"Travel time: {time:.2f} minutes\n"
                for j in range(len(path)-1):
                    segment_time = self.G[path[j]][path[j+1]]['weight']
                    result += f"  {path[j]} → {path[j+1]}: {segment_time:.2f} min\n"
                result += "\n"
        else:
            result += "No routes found between selected nodes.\n"
        self.result_queue.put(result)

    def check_queue(self):
        try:
            result = self.result_queue.get_nowait()
            self.result_text.insert(tk.END, result)
            self.find_button.config(state='normal')
            if self.model is not None:
                self.train_button.config(state='normal')
        except queue.Empty:
            self.root.after(100, self.check_queue)

    def on_closing(self):
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

def main():
    G = create_road_network('traffic_network2.csv')
    site_ids = list(G.nodes())
    print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    if G.number_of_nodes() != 41:
        print("Warning: Expected 41 SCATS sites, but found", G.number_of_nodes())
    ui = RouteFinderUI(G, site_ids)
    ui.run()

if __name__ == "__main__":
    main()