import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from pol_attack import demonstrate_pol_concatenation_attack
import utils

def calculate_parameter_distances(concat_dir, architecture, save_freq=100):
    """Calculates all parameter distances between consecutive checkpoints"""
    checkpoint_files = [f for f in os.listdir(concat_dir) if f.startswith("model_step_")]
    step_nums = sorted([int(cp.split('_')[-1]) for cp in checkpoint_files])
    
    distances = []
    for i in range(len(step_nums)-1):
        model1_path = os.path.join(concat_dir, f"model_step_{step_nums[i]}")
        model2_path = os.path.join(concat_dir, f"model_step_{step_nums[i+1]}")
        
        distance = utils.parameter_distance(model1_path, model2_path, 
                                           order=['2'], architecture=architecture)[0]
        distances.append((step_nums[i], step_nums[i+1], distance))
    
    return distances

def random_sampling_verification(distances, num_samples, threshold):
    """Simulates random sampling verification with iteration tracking"""
    selected_indices = random.sample(range(len(distances)), num_samples)
    random.shuffle(selected_indices)  # Randomize the checking order
    
    # Check one by one and record when we first detect the discontinuity
    for i, idx in enumerate(selected_indices):
        if distances[idx][2] > threshold:
            # Found discontinuity at iteration i+1
            return True, i+1, distances[idx][2]
    
    # No discontinuity found
    return False, num_samples, 0

def top_q_verification(distances, q, threshold):
    """Simulate top-Q verification"""
    # Sort distances by magnitude (descending)
    sorted_distances = sorted(distances, key=lambda x: x[2], reverse=True)
    
    # Take top q distances
    top_q_distances = [d[2] for d in sorted_distances[:q]]
    
    # Check if any distance exceeds threshold
    detected = any(d > threshold for d in top_q_distances)
    return detected, top_q_distances

def empirical_comparison(num_trials=100):
    """Run empirical comparison between random sampling and top-Q verification"""
    # First create the spoofed PoL
    print("Creating spoofed PoL for verification demonstration...")
    concat_dir = demonstrate_pol_concatenation_attack(
        model_name='Simple_Conv',
        dataset='MNIST',
        initial_steps=500,
        batch_size=128,
        lr=0.01,
        exp_id='verification_demo'
    )
    
    # Get architecture
    import model as custom_model
    architecture = custom_model.Simple_Conv
    
    # Calculate all parameter distances
    print("\nCalculating parameter distances between all checkpoints...")
    distances = calculate_parameter_distances(concat_dir, architecture)
    
    # Set threshold to 10x average normal update size
    normal_updates = [d[2] for d in distances if d[2] < max([d[2] for d in distances]) / 10]
    threshold = np.mean(normal_updates) * 10
    
    print(f"Found {len(distances)} checkpoint pairs")
    print(f"Maximum distance (at discontinuity): {max([d[2] for d in distances]):.4f}")
    print(f"Average normal update distance: {np.mean(normal_updates):.4f}")
    print(f"Threshold set to: {threshold:.4f}")
    
    # Number of samples to check (equivalent to E in the paper)
    num_samples = min(10, len(distances))
    
    # Compare approaches over multiple trials
    random_detection_rate = 0
    top_q_detection_rate = 0
    
    random_iterations = []
    
    print(f"\nRunning {num_trials} trials to compare verification methods...")
    for i in range(num_trials):
        # Random sampling (now returns iterations required)
        random_detected, iterations_needed, max_distance = random_sampling_verification(
            distances, num_samples, threshold)
            
        if random_detected:
            random_detection_rate += 1
            random_iterations.append(iterations_needed)
            
        # Top-Q always checks the largest change first
        top_q_detected, _ = top_q_verification(distances, 1, threshold)
        if top_q_detected:
            top_q_detection_rate += 1
    
    # Calculate average iterations for random sampling when successful
    avg_iterations = np.mean(random_iterations) if random_iterations else "N/A"
    
    # Add a histogram of iterations needed
    if random_iterations:
        plt.figure(figsize=(8, 5))
        plt.hist(random_iterations, bins=range(1, num_samples+2), 
                 align='left', alpha=0.7, rwidth=0.8)
        plt.title('Iterations Required for Random Sampling to Detect Discontinuity')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Frequency')
        plt.xticks(range(1, num_samples+1))
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(os.path.join(concat_dir, 'random_iterations_histogram.png'))

    # Calculate empirical detection rates
    random_detection_rate /= num_trials
    top_q_detection_rate /= num_trials
    
    # Calculate theoretical detection rate for random sampling
    theoretical_random_rate = min(num_samples / len(distances), 1.0)
    
    # print("\n=== Verification Results ===")
    # print(f"Total checkpoint pairs: {len(distances)}")
    # print(f"Number of samples checked per trial: {num_samples}")
    # print(f"Top-Q verification detection rate: {top_q_detection_rate:.2%}")
    # print(f"Random sampling detection rate: {random_detection_rate:.2%}")
    print("\n=== Verification Results ===")
    print(f"Total checkpoint pairs: {len(distances)}")
    print(f"Number of samples checked per trial: {num_samples}")
    print(f"Top-Q verification detection rate: {top_q_detection_rate:.2%}")
    print(f"Random sampling detection rate: {random_detection_rate:.2%}")
    print(f"Average iterations required for random detection: {avg_iterations}")
    print(f"Theoretical random detection rate: {theoretical_random_rate:.2%}")
    
    # Visualize the results
    plt.figure(figsize=(10, 6))
    distances_only = [d[2] for d in distances]
    plt.stem(range(len(distances)), distances_only, 'b-', 
             markerfmt='bo', label='Parameter distances')
    plt.axhline(y=threshold, color='r', linestyle='--', 
                label=f'Threshold: {threshold:.4f}')
    
    # Mark discontinuity
    discontinuity_idx = np.argmax(distances_only)
    plt.annotate('Discontinuity', 
                 xy=(discontinuity_idx, distances_only[discontinuity_idx]),
                 xytext=(discontinuity_idx, distances_only[discontinuity_idx]*0.8),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 horizontalalignment='center')
    
    plt.title('Parameter Distances Between Consecutive Checkpoints')
    plt.xlabel('Checkpoint Pair Index')
    plt.ylabel('L2 Distance')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(concat_dir, 'verification_comparison.png'))
    
    # Return results
    return {
        'top_q_rate': top_q_detection_rate,
        'random_rate': random_detection_rate,
        'theoretical_random_rate': theoretical_random_rate,
        'distances': distances,
        'threshold': threshold
    }

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    results = empirical_comparison(num_trials=100)
    print("\nVerification demonstration complete.")