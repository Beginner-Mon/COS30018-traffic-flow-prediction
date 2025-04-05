import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime, timedelta


def load_data(predictions_file, ground_truth_file, uncertainty_file=None):
    """Load predictions, ground truth, and uncertainty data from specified files."""
    predictions = np.load(predictions_file)
    ground_truth = np.load(ground_truth_file)
    uncertainty = np.load(uncertainty_file) if uncertainty_file else None
    return predictions, ground_truth, uncertainty


def plot_comparison(predictions, ground_truth, uncertainty=None, horizon=4, node_idx=None,
                    start_timestamp="2006-10-01 00:00", time_interval=15,
                    mae_by_horizon=None, rmse_by_horizon=None, output_dir='results'):
    """
    Plot predictions against ground truth with uncertainty and error metrics.

    Args:
        predictions: numpy array of shape (samples, horizon, nodes, num_directions)
        ground_truth: numpy array of shape (samples, horizon, nodes, num_directions)
        uncertainty: numpy array of shape (samples, horizon, nodes, num_directions), optional
        horizon: number of prediction horizons
        node_idx: index of the node (SCATS site) to plot; if None, aggregate across nodes
        start_timestamp: starting timestamp for the data (str, format: "YYYY-MM-DD HH:MM")
        time_interval: time interval between samples in minutes (default: 15)
        mae_by_horizon: list of MAE values for each horizon
        rmse_by_horizon: list of RMSE values for each horizon
        output_dir: directory to save the plot
    """
    # Aggregate across directions (sum flows for each node)
    predictions_agg = predictions.sum(axis=-1)  # Shape: (samples, horizon, nodes)
    ground_truth_agg = ground_truth.sum(axis=-1)  # Shape: (samples, horizon, nodes)
    if uncertainty is not None:
        uncertainty_agg = uncertainty.sum(axis=-1)  # Shape: (samples, horizon, nodes)
    else:
        uncertainty_agg = None

    # If node_idx is specified, select data for that node; otherwise, average across nodes
    if node_idx is not None:
        pred_data = predictions_agg[:, :, node_idx]
        truth_data = ground_truth_agg[:, :, node_idx]
        uncert_data = uncertainty_agg[:, :, node_idx] if uncertainty_agg is not None else None
        title_suffix = f" (SCATS {node_idx})"
    else:
        pred_data = predictions_agg.mean(axis=2)  # Average across nodes
        truth_data = ground_truth_agg.mean(axis=2)  # Average across nodes
        uncert_data = uncertainty_agg.mean(axis=2) if uncertainty_agg is not None else None
        title_suffix = " (Averaged Across Nodes)"

    # Generate timestamps for the x-axis
    start_time = pd.to_datetime(start_timestamp, format="%Y-%m-%d %H:%M")
    num_samples = pred_data.shape[0]
    timestamps = [start_time + timedelta(minutes=i * time_interval) for i in range(num_samples)]
    timestamp_labels = [ts.strftime("%Y-%m-%d %H:%M") for ts in timestamps]

    # Plotting
    plt.figure(figsize=(15, 3 * horizon))

    for h in range(horizon):
        plt.subplot(horizon, 1, h + 1)

        # Plot ground truth
        plt.plot(timestamps, truth_data[:, h], label='Ground Truth', color='orange', alpha=0.7)

        # Plot predictions with uncertainty
        plt.plot(timestamps, pred_data[:, h], label='Predictions', color='blue', alpha=0.7)
        if uncert_data is not None:
            plt.fill_between(timestamps,
                             pred_data[:, h] - uncert_data[:, h],
                             pred_data[:, h] + uncert_data[:, h],
                             color='blue', alpha=0.2, label='Uncertainty (±1 STD)')

        # Add title with metrics
        title = f'Horizon {h + 1} ({(h + 1) * 15} min)'
        if mae_by_horizon and rmse_by_horizon:
            title += f' - MAE: {mae_by_horizon[h]:.4f}, RMSE: {rmse_by_horizon[h]:.4f}'
        plt.title(title + title_suffix)

        plt.xlabel('Timestamp')
        plt.ylabel('Flow (vehicles/15min)')
        plt.legend()

        # Rotate x-axis labels for readability
        plt.xticks(rotation=45)

        # Show only a subset of timestamps to avoid clutter
        plt.gca().set_xticks(timestamps[::max(1, num_samples // 10)])
        plt.gca().set_xticklabels(timestamp_labels[::max(1, num_samples // 10)])

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'flow_comparison{"_node_" + str(node_idx) if node_idx is not None else ""}.png'))
    plt.show()


def main():
    # Specify the directory containing the results
    output_dir = ''

    # Load predictions, ground truth, and uncertainty
    predictions_file = os.path.join(output_dir, 'predictions_best.npy')  # Update with your timestamp
    ground_truth_file = os.path.join(output_dir, 'ground_truth_best.npy')
    uncertainty_file = os.path.join(output_dir, 'uncertainty_best.npy')

    predictions, ground_truth, uncertainty = load_data(predictions_file, ground_truth_file, uncertainty_file)

    # Assuming the horizon is the second dimension of predictions
    horizon = predictions.shape[1]

    # Metrics from your output
    mae_by_horizon = [0.0829, 0.0791, 0.0824, 0.0883]
    rmse_by_horizon = [0.1944, 0.1958, 0.2004, 0.2052]

    # Plot comparison (average across nodes)
    plot_comparison(predictions, ground_truth, uncertainty, horizon=horizon, node_idx=None,
                    start_timestamp="2006-10-01 00:00", time_interval=15,
                    mae_by_horizon=mae_by_horizon, rmse_by_horizon=rmse_by_horizon,
                    output_dir=output_dir)

    # Plot for a specific node (e.g., node 0)
    plot_comparison(predictions, ground_truth, uncertainty, horizon=horizon, node_idx=0,
                    start_timestamp="2006-10-01 00:00", time_interval=15,
                    mae_by_horizon=mae_by_horizon, rmse_by_horizon=rmse_by_horizon,
                    output_dir=output_dir)


if __name__ == "__main__":
    main()