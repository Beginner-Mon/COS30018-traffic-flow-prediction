import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(predictions_file, ground_truth_file):
    """Load predictions and ground truth data from specified files."""
    predictions = np.load(predictions_file)
    ground_truth = np.load(ground_truth_file)
    return predictions, ground_truth

def plot_comparison(predictions, ground_truth, horizon):
    """Plot predictions against ground truth."""
    plt.figure(figsize=(12, 6))
    
    for h in range(horizon):
        plt.subplot(horizon, 1, h + 1)
        plt.plot(predictions[:, h, :].flatten(), label='Predictions', alpha=0.7)
        plt.plot(ground_truth[:, h, :].flatten(), label='Ground Truth', alpha=0.7)
        plt.title(f'Horizon {h + 1}')
        plt.xlabel('Time Steps')
        plt.ylabel('Flow')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Specify the directory containing the results
    output_dir = ''
    
    # Load predictions and ground truth
    predictions_file = os.path.join(output_dir, 'predictions_11.npy')
    ground_truth_file = os.path.join(output_dir, 'ground_truth_11.npy')
    
    predictions, ground_truth = load_data(predictions_file, ground_truth_file)
    
    # Assuming the horizon is the second dimension of predictions
    horizon = predictions.shape[1]
    
    # Plot comparison
    plot_comparison(predictions, ground_truth, horizon)

if __name__ == "__main__":
    main()