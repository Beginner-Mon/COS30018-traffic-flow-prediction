import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def load_latest_results(results_dir='./results'):
    """Find and load the latest prediction/truth pair"""
    # Find all prediction and truth files
    pred_files = sorted(glob.glob(os.path.join(results_dir, 'predictions_*.npy')),
                        key=os.path.getmtime)
    truth_files = sorted(glob.glob(os.path.join(results_dir, 'ground_truth_*.npy')),
                         key=os.getmtime)

    if not pred_files or not truth_files:
        raise FileNotFoundError("No result files found in the directory")

    # Find matching timestamp pairs
    valid_pairs = []
    for pred_file in pred_files:
        timestamp = os.path.basename(pred_file).split('_')[1]
        matching_truth = [f for f in truth_files if timestamp in f]
        if matching_truth:
            valid_pairs.append((pred_file, matching_truth[0]))

    if not valid_pairs:
        raise ValueError("No matching prediction-truth pairs found")

    # Get most recent pair
    latest_pair = valid_pairs[-1]
    print(f"Loading: {latest_pair[0]}\n       {latest_pair[1]}")

    return np.load(latest_pair[0]), np.load(latest_pair[1]))

    def plot_comparison(predictions, ground_truth, node_idx=0, horizon_step=0, save_path=None))

        :
    """
    Plot predictions vs ground truth for a specific node and prediction horizon
    Args:
        predictions: 3D array (num_samples, horizon, num_nodes)
        ground_truth: 3D array (num_samples, horizon, num_nodes)
        node_idx: Which node to visualize
        horizon_step: Which prediction step to visualize (0 = first predicted step)
        save_path: Where to save the plot (None shows interactive plot)
    """
    # Extract data for selected node and horizon
    preds = predictions[:, horizon_step, node_idx]
    truths = ground_truth[:, horizon_step, node_idx]

    # Calculate metrics
    mae = np.mean(np.abs(preds - truths))
    rmse = np.sqrt(np.mean((preds - truths) ** 2))

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(truths, label='Ground Truth', color='#1f77b4', linewidth=2)
    plt.plot(preds, label='Predictions', color='#ff7f0e', linestyle='--', linewidth=2)

    # Formatting
    plt.title(f"Traffic Flow Prediction vs Actual\n"
    f"Node {node_idx} | Horizon Step {horizon_step + 1}\n"
    f"MAE: {mae:.2f} | RMSE: {rmse:.2f}", fontsize = 14)
    plt.xlabel("Time Steps (15-min intervals)", fontsize=12)
    plt.ylabel("Vehicle Flow (veh/h)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {save_path}")
    else:
    plt.show()


def main(results_dir='./results'):
    # Load data
    predictions, ground_truth = load_latest_results(results_dir)

    # Create output directory if needed
    os.makedirs(results_dir, exist_ok=True)

    # Generate comparison plots
    for node in [0, 4, 10]:  # Example nodes to plot
        save_path = os.path.join(results_dir, f'node_{node}_comparison.png')
        plot_comparison(
            predictions,
            ground_truth,
            node_idx=node,
            horizon_step=0,  # First prediction step
            save_path=save_path
        )


if __name__ == "__main__":
    main()