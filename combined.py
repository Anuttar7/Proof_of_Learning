import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import OrderedDict

import model as custom_model
import utils
from train import train, validate

def clear_directories(dataset, exp_ids):
    """Clear existing directories for a fresh run"""
    base_dir = "proof"
    for exp_id in exp_ids:
        dir_path = os.path.join(base_dir, f"{dataset}_{exp_id}")
        if os.path.exists(dir_path):
            print(f"Removing existing directory: {dir_path}")
            import shutil
            shutil.rmtree(dir_path)

def train_with_large_lr_spikes(
    model_name="Simple_Conv",
    dataset="MNIST",
    epochs=5,
    batch_size=128,
    normal_lr=0.01,
    large_lr=1.0,  # 100x larger than normal
    q_value=3,     # Adversary targets bypassing top-3 verification
    save_freq=100,
    exp_id="large_lr_attack"
):
    """
    Train a model with periodic large learning rate spikes to generate Q+1 large but valid updates.
    This simulates an adversary trying to bypass top-Q verification by making multiple large updates.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create proof directory if it doesn't exist
    if not os.path.exists("proof"):
        os.makedirs("proof")
    
    # Clear existing directories
    clear_directories(dataset, [exp_id, "normal_training"])
    
    # Get model architecture
    try:
        architecture = getattr(custom_model, model_name)
    except AttributeError:
        raise ValueError(f"Model {model_name} not found. Try 'Simple_Conv' or 'resnet20'.")
    
    # Load dataset
    trainset = utils.load_dataset(dataset, True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Create model and optimizer
    model = architecture().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=normal_lr)
    
    # Save directory
    save_dir = os.path.join("proof", f"{dataset}_{exp_id}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save initial model
    state = {'model': model.state_dict()}
    torch.save(state, os.path.join(save_dir, f"model_step_0"))
    
    # # Copy the hash.txt from the normal training (will be created below)
    # from train import get_hash
    # hash_value = get_hash(dataset)
    # with open(os.path.join(save_dir, "hash.txt"), "w") as f:
    #     f.write(hash_value)
    
    # Track parameter distances
    distance_log = []
    lr_log = []
    accuracy_log = []
    
    # Track training indices
    all_indices = []
    
    # Number of batches where we'll use large learning rate 
    # We want q_value+1 large updates to ensure at least one isn't caught by top-Q
    num_large_lr_points = q_value + 1
    
    # Spread the large learning rate points across training
    total_batches = len(trainloader) * epochs
    large_lr_batches = set(np.linspace(0, total_batches-1, num_large_lr_points, dtype=int))
    
    print(f"Training with {num_large_lr_points} large learning rate spikes")
    print(f"Large learning rate will be used at batches: {sorted(large_lr_batches)}")
    
    # Training loop
    batch_idx = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Choose learning rate for this batch
            if batch_idx in large_lr_batches:
                # Use large learning rate for this batch
                for param_group in optimizer.param_groups:
                    param_group['lr'] = large_lr
                print(f"Epoch {epoch+1}, Batch {i+1}: Using large learning rate {large_lr}")
            else:
                # Use normal learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = normal_lr
            
            # Get model state before update for distance calculation
            prev_params = utils.get_parameters(model, numpy=True)
            
            # Standard training step
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate parameter distance after update
            curr_params = utils.get_parameters(model, numpy=True)
            update_dist = np.linalg.norm(curr_params - prev_params)
            
            # Log the distance and current learning rate
            distance_log.append(update_dist)
            lr_log.append(optimizer.param_groups[0]['lr'])
            
            # Log the indices used for training
            all_indices.extend(range(i*batch_size, min((i+1)*batch_size, len(trainset))))
            
            # Track loss
            running_loss += loss.item()
            
            # Save model checkpoint at regular intervals
            if batch_idx % save_freq == 0 and batch_idx > 0:
                state = {'model': model.state_dict()}
                torch.save(state, os.path.join(save_dir, f"model_step_{batch_idx}"))
                
                # Calculate accuracy
                accuracy = validate(model=model, dataset=dataset)
                accuracy_log.append((batch_idx, accuracy))
                print(f"Batch {batch_idx}: Accuracy = {accuracy:.2f}%")
            
            batch_idx += 1
            
            # Print statistics
            if i % 100 == 99:
                print(f"Epoch {epoch+1}, Batch {i+1}: Loss = {running_loss/100:.3f}")
                running_loss = 0.0
    
    # Save final model
    state = {'model': model.state_dict()}
    torch.save(state, os.path.join(save_dir, f"model_step_{batch_idx}"))
    
    # Calculate final accuracy
    accuracy = validate(model=model, dataset=dataset)
    accuracy_log.append((batch_idx, accuracy))
    print(f"Final accuracy: {accuracy:.2f}%")
    
    # Save the training indices
    all_indices = np.array(list(set(all_indices)))  # Remove duplicates
    np.save(os.path.join(save_dir, "indices.npy"), all_indices)
    
    # Save logs
    np.savez(os.path.join(save_dir, "training_logs.npz"),
             distances=np.array(distance_log),
             learning_rates=np.array(lr_log),
             accuracies=np.array(accuracy_log))
    
    # Plot parameter distance distribution
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(distance_log)
    plt.yscale('log')
    plt.title('Parameter Distance per Batch')
    plt.xlabel('Batch')
    plt.ylabel('L2 Distance')
    plt.grid(True)
    
    # Mark the large learning rate batches
    for batch in sorted(large_lr_batches):
        if batch < len(distance_log):
            plt.axvline(x=batch, color='r', linestyle='--', alpha=0.5)
    
    plt.subplot(2, 1, 2)
    plt.plot(lr_log)
    plt.yscale('log')
    plt.title('Learning Rate per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "parameter_distances.png"))
    
    # Also train a model normally for comparison
    print("\nTraining a model with normal learning rate for comparison...")
    normal_exp_id = "normal_training"
    normal_model = train(normal_lr, batch_size, epochs, dataset, architecture, exp_id=normal_exp_id,
                         save_freq=save_freq)
    
    return save_dir, f"proof/{dataset}_{normal_exp_id}"

def verify_with_countermeasures(attack_dir, normal_dir, q_value=3):
    """
    Test the effectiveness of the three proposed countermeasures:
    1. Using a larger Q value
    2. Randomly verifying additional updates
    3. Checking model performance
    """
    print("\n===== Testing Countermeasures =====")
    
    # Load training logs
    logs = np.load(os.path.join(attack_dir, "training_logs.npz"))
    distances = logs['distances']
    learning_rates = logs['learning_rates']
    accuracies = logs['accuracies']
    
    # 1. Test increasing Q value
    print("\n1. Testing different Q values:")
    # Sort distances in descending order
    sorted_indices = np.argsort(distances)[::-1]
    
    for test_q in [1, 3, 5, 10]:
        # Get the top-Q largest parameter updates
        top_q_indices = sorted_indices[:test_q]
        top_q_distances = distances[top_q_indices]
        
        # Check if all large learning rate batches are caught
        large_lr_indices = np.where(learning_rates > learning_rates.min())[0]
        large_lr_caught = sum(idx in top_q_indices for idx in large_lr_indices)
        
        print(f"Using Q={test_q}:")
        print(f"  - Caught {large_lr_caught} out of {len(large_lr_indices)} large learning rate updates")
        print(f"  - Smallest distance in top-{test_q}: {np.min(top_q_distances):.6f}")
        
        if large_lr_caught == len(large_lr_indices):
            print(f"  - SUCCESS: All large learning rate updates detected with Q={test_q}")
        else:
            print(f"  - FAILURE: {len(large_lr_indices) - large_lr_caught} large updates escaped detection")
    
    # 2. Test random verification + top-Q
    print("\n2. Testing top-Q + random sampling:")
    for random_samples in [5, 10, 20]:
        # Combine top-Q with random sampling
        top_q_indices = sorted_indices[:q_value]  # Standard top-Q
        
        # Randomly sample from remaining indices
        remaining_indices = [i for i in range(len(distances)) if i not in top_q_indices]
        random_indices = np.random.choice(remaining_indices, 
                                         size=min(random_samples, len(remaining_indices)), 
                                         replace=False)
        
        # Combine top-Q and random samples
        combined_indices = np.concatenate([top_q_indices, random_indices])
        
        # Check if all large learning rate batches are caught
        large_lr_caught = sum(idx in combined_indices for idx in large_lr_indices)
        
        print(f"Using Q={q_value} + {random_samples} random samples:")
        print(f"  - Caught {large_lr_caught} out of {len(large_lr_indices)} large learning rate updates")
        
        if large_lr_caught == len(large_lr_indices):
            print(f"  - SUCCESS: All large learning rate updates detected")
        else:
            print(f"  - FAILURE: {len(large_lr_indices) - large_lr_caught} large updates escaped detection")
    
    # 3. Test performance-based detection
    print("\n3. Testing performance-based detection:")
    
    # Calculate performance drops
    acc_values = accuracies[:, 1]  # Extract accuracy values
    acc_steps = accuracies[:, 0]   # Extract corresponding steps
    
    # Find performance drops
    performance_drops = []
    for i in range(1, len(acc_values)):
        if acc_values[i] < acc_values[i-1]:
            drop = acc_values[i-1] - acc_values[i]
            step = acc_steps[i]
            performance_drops.append((step, drop))
    
    # Sort drops by magnitude
    performance_drops.sort(key=lambda x: x[1], reverse=True)
    
    # Check if performance drops correlate with large learning rate batches
    if performance_drops:
        print(f"Largest performance drops:")
        for step, drop in performance_drops[:5]:
            # Find the closest large learning rate batch
            closest_large_lr = min(large_lr_indices, key=lambda x: abs(x - step))
            distance = abs(closest_large_lr - step)
            
            print(f"  - Step {step}: Drop of {drop:.2f}% (Distance to closest large LR: {distance} steps)")
            
        # Calculate correlation between performance drops and large learning rates
        drop_steps = [d[0] for d in performance_drops]
        
        # Check if top performance drops would catch large learning rate batches
        detected = 0
        threshold = 100  # Steps threshold for considering a drop related to large LR
        for large_lr_idx in large_lr_indices:
            if any(abs(large_lr_idx - step) < threshold for step in drop_steps[:3]):
                detected += 1
                
        print(f"\nPerformance-based detection results:")
        print(f"  - Top 3 performance drops would detect {detected} out of {len(large_lr_indices)} large LR batches")
        if detected == len(large_lr_indices):
            print("  - SUCCESS: All large learning rate batches detected via performance drops")
        else:
            print(f"  - PARTIAL SUCCESS: {detected}/{len(large_lr_indices)} detected via performance drops")
    else:
        print("No performance drops detected")
        
    # 4. Test combined countermeasure: moderate Q + performance monitoring
    print("\n4. Testing combined approach (moderate Q + performance monitoring):")

    # Use a moderate Q value (not as large as needed for complete detection)
    moderate_q = min(5, q_value + 2)  # Use either 5 or q_value+2, whichever is smaller
    moderate_q_indices = sorted_indices[:moderate_q]

    # Check which large updates are caught by moderate Q
    param_detected = [idx for idx in large_lr_indices if idx in moderate_q_indices]
    param_missed = [idx for idx in large_lr_indices if idx not in moderate_q_indices]

    print(f"With parameter-based detection (Q={moderate_q}):")
    print(f"  - Detected: {len(param_detected)} of {len(large_lr_indices)} large updates")
    print(f"  - Missed: {len(param_missed)} large updates")

    # Performance-based detection (reuse the performance drops from part 3)
    # Take top 3 performance drops
    top_perf_steps = [int(d[0]) for d in performance_drops[:3]]

    # Find which large LR batches are near performance drops
    perf_threshold = 100  # Steps threshold
    perf_detected = []

    for large_lr_idx in large_lr_indices:
        if any(abs(large_lr_idx - step) < perf_threshold for step in top_perf_steps):
            perf_detected.append(large_lr_idx)

    perf_missed = [idx for idx in large_lr_indices if idx not in perf_detected]

    print(f"\nWith performance-based detection (top 3 drops):")
    print(f"  - Detected: {len(perf_detected)} of {len(large_lr_indices)} large updates")
    print(f"  - Missed: {len(perf_missed)} large updates") 

    # Combine parameter and performance detection
    combined_detected = list(set(param_detected + perf_detected))
    combined_missed = [idx for idx in large_lr_indices if idx not in combined_detected]

    print(f"\nWith COMBINED approach:")
    print(f"  - Detected: {len(combined_detected)} of {len(large_lr_indices)} large updates")

    if len(combined_missed) == 0:
        print(f"  - SUCCESS: All large learning rate updates detected with combined approach")
    else:
        print(f"  - PARTIAL SUCCESS: {len(combined_detected)}/{len(large_lr_indices)} detected")

    # Calculate verification cost
    param_cost = moderate_q  # Q verifications
    perf_cost = len(top_perf_steps)  # Performance drop verifications
    total_cost = param_cost + perf_cost

    print(f"\nVerification cost comparison:")
    print(f"  - Large Q only (Q=10): 10 verifications")
    print(f"  - Combined approach: {total_cost} verifications ({moderate_q} parameter + {perf_cost} performance)")

    # Visualize the combined approach
    plt.figure(figsize=(12, 8))

    # Plot 1: All parameter distances with detected updates marked
    plt.subplot(2, 1, 1)
    plt.plot(distances, 'b-', alpha=0.5, label='Parameter Distance')
    plt.yscale('log')

    # Mark all large learning rate points
    for idx in large_lr_indices:
        plt.axvline(x=idx, color='r', linestyle='--', alpha=0.3)

    # Mark param-detected points
    for idx in param_detected:
        plt.scatter(idx, distances[idx], color='g', s=100, zorder=5, marker='o')

    # Mark perf-detected points
    for idx in perf_detected:
        if idx not in param_detected:  # Only highlight those not already caught by params
            plt.scatter(idx, distances[idx], color='purple', s=100, zorder=5, marker='x')

    plt.title('Combined Detection Approach')
    plt.xlabel('Batch')
    plt.ylabel('Parameter Distance (L2)')
    plt.grid(True)

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='r', linestyle='--', label='Large LR Batch'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, 
            label=f'Detected by Top-{moderate_q}'),
        Line2D([0], [0], marker='x', color='purple', markersize=10, 
            label='Detected by Performance')
    ]
    plt.legend(handles=legend_elements)

    # Plot 2: Performance drops
    plt.subplot(2, 1, 2)
    plt.plot(accuracies[:, 0], accuracies[:, 1], 'o-', label='Model Accuracy')

    # Mark top performance drops
    for step in top_perf_steps:
        plt.axvline(x=step, color='purple', linestyle='--', alpha=0.7)
        closest_idx = np.argmin(np.abs(accuracies[:, 0] - step))
        plt.scatter(accuracies[closest_idx, 0], accuracies[closest_idx, 1], 
                color='purple', s=150, zorder=5, marker='x')

    plt.title('Accuracy with Performance Drops Highlighted')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(attack_dir, "combined_countermeasure.png"))



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Demonstrate large learning rate attack and countermeasures')
    parser.add_argument('--model', type=str, default='Simple_Conv', 
                        help='Model architecture (e.g., Simple_Conv, resnet20)')
    parser.add_argument('--dataset', type=str, default='MNIST', 
                        help='Dataset (MNIST or CIFAR10)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--normal-lr', type=float, default=0.01,
                        help='Normal learning rate')
    parser.add_argument('--large-lr', type=float, default=1.0,
                        help='Large learning rate for attack')
    parser.add_argument('--q-value', type=int, default=3,
                        help='Number of top updates to verify (Q)')
    
    args = parser.parse_args()
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    attack_dir, normal_dir = train_with_large_lr_spikes(
        model_name=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        normal_lr=args.normal_lr,
        large_lr=args.large_lr,
        q_value=args.q_value
    )
    
    results = verify_with_countermeasures(attack_dir, normal_dir, q_value=args.q_value)
    
    print("\nExperiment complete! Results saved to:", attack_dir)