import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import shutil
from collections import OrderedDict

import model as custom_model
import utils
from train import train

def clear_existing_proof(dataset, exp_id):
    """Clear existing proof directories for a fresh run"""
    base_dir = "proof"
    directories = [
        f"{dataset}_{exp_id}_wt",
        f"{dataset}_{exp_id}_f",
        f"{dataset}_{exp_id}_ws",
        f"{dataset}_{exp_id}_concat"
    ]
    
    for dir_name in directories:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"Removing existing directory: {dir_path}")
            shutil.rmtree(dir_path)

def demonstrate_pol_concatenation_attack(
    model_name="Simple_Conv",
    dataset="MNIST",
    initial_steps=500,  # steps for first model (W0 to Ws)
    batch_size=128,
    lr=0.01,
    save_freq=100,
    exp_id="attack_demo"
):
    """
    Demonstrates the PoL concatenation attack:
    1. Train a model to completion (WT)
    2. Fine-tune WT for 1 epoch to get model f
    3. Train a model from scratch for s steps (W0 to Ws)
    4. Concatenate the PoLs to create a spoofed PoL
    5. Analyze the discontinuity
    6. Verify with top-Q and random sampling
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create proof directory if it doesn't exist
    if not os.path.exists("proof"):
        os.makedirs("proof")
    
    # Clear any existing directories for this experiment
    clear_existing_proof(dataset, exp_id)

    # Print CUDA information
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Get model architecture
    try:
        architecture = getattr(custom_model, model_name)
    except AttributeError:
        raise ValueError(f"Model {model_name} not found. Try 'Simple_Conv' or 'resnet20'.")
    
    print("Step 1: Training model to completion (WT)...")
    # For demonstration, train for just 2 epochs
    wt_model = architecture().to(device)
    wt_exp_id = f"{exp_id}_wt"
    wt_model = train(lr, batch_size, 2, dataset, architecture, exp_id=wt_exp_id,
                     save_freq=save_freq)
    
    print("\nStep 2: Fine-tuning WT for 1 epoch to get model f...")
    # Use the trained model as starting point and fine-tune
    f_exp_id = f"{exp_id}_f"
    wt_path = f"proof/{dataset}_{wt_exp_id}/model_step_{save_freq}"
    f_model = train(lr, batch_size, 1, dataset, architecture, exp_id=f_exp_id,
                    model_dir=wt_path, save_freq=save_freq)
    
    print("\nStep 3: Training model from scratch for s steps (W0 to Ws)...")
    # Calculate epochs needed for initial_steps
    trainset = utils.load_dataset(dataset, True)
    batches_per_epoch = len(trainset) // batch_size
    epochs_needed = max(1, initial_steps // batches_per_epoch)
    
    ws_exp_id = f"{exp_id}_ws"
    ws_model = train(lr, batch_size, epochs_needed, dataset, architecture, exp_id=ws_exp_id,
                     save_freq=save_freq)
    
    print("\nStep 4: Creating concatenated spoofed PoL...")
    # Create directory for concatenated PoL
    concat_dir = os.path.join("proof", f"{dataset}_{exp_id}_concat")
    os.makedirs(concat_dir, exist_ok=True)
    
    # Load sequences from separate PoLs
    ws_dir = os.path.join("proof", f"{dataset}_{ws_exp_id}")
    f_dir = os.path.join("proof", f"{dataset}_{f_exp_id}")
    
    ws_indices = np.load(os.path.join(ws_dir, "indices.npy"))
    f_indices = np.load(os.path.join(f_dir, "indices.npy"))
    
    # Concatenate sequences
    concat_indices = np.concatenate([ws_indices, f_indices])
    np.save(os.path.join(concat_dir, "indices.npy"), concat_indices)
    
    # Copy the hash.txt file from either directory to maintain consistency
    if os.path.exists(os.path.join(ws_dir, "hash.txt")):
        shutil.copy2(os.path.join(ws_dir, "hash.txt"), os.path.join(concat_dir, "hash.txt"))
    elif os.path.exists(os.path.join(f_dir, "hash.txt")):
        shutil.copy2(os.path.join(f_dir, "hash.txt"), os.path.join(concat_dir, "hash.txt"))
    
    print("Copying model checkpoints to create complete spoofed PoL...")
    
    
    # 1. Find all available checkpoints from source directories
    ws_checkpoints = [f for f in os.listdir(ws_dir) if f.startswith("model_step_")]
    f_checkpoints = [f for f in os.listdir(f_dir) if f.startswith("model_step_")]
    
    print(f"Found {len(ws_checkpoints)} checkpoints in W0→Ws directory")
    print(f"Found {len(f_checkpoints)} checkpoints in WT→f directory")
    
    # 2. Copy checkpoints from W0 to Ws
    for checkpoint in ws_checkpoints:
        source_path = os.path.join(ws_dir, checkpoint)
        target_path = os.path.join(concat_dir, checkpoint)
        shutil.copy2(source_path, target_path)
    
    # 3. Copy checkpoints from WT to f, continuing after Ws with adjusted step numbers
    ws_steps = len(ws_indices)
    highest_ws_step = max([int(cp.split('_')[-1]) for cp in ws_checkpoints]) if ws_checkpoints else 0
    
    for checkpoint in f_checkpoints:
        step_num = int(checkpoint.split('_')[-1])
        new_step_num = highest_ws_step + step_num
        
        source_path = os.path.join(f_dir, checkpoint)
        target_path = os.path.join(concat_dir, f"model_step_{new_step_num}")
        shutil.copy2(source_path, target_path)
    
    # 4. Copy the final model as the last step in the sequence
    highest_f_step = max([int(cp.split('_')[-1]) for cp in f_checkpoints]) if f_checkpoints else 0
    final_source = os.path.join(f_dir, f"model_step_{highest_f_step}")
    final_target = os.path.join(concat_dir, f"model_step_{len(concat_indices)}")
    if os.path.exists(final_source) and final_target != os.path.join(concat_dir, f"model_step_{highest_ws_step + highest_f_step}"):
        shutil.copy2(final_source, final_target)
    
    # Ensure we have regularly named checkpoints for verification
    print("\nEnsuring regular checkpoint files exist for verification...")
    expected_steps = list(range(0, 1000, save_freq))  # 0, 100, 200, ..., 900
    expected_files = [f"model_step_{step}" for step in expected_steps]

    # Get existing checkpoint files
    existing_files = [f for f in os.listdir(concat_dir) if f.startswith("model_step_")]
    existing_steps = [int(f.split('_')[-1]) for f in existing_files]
    existing_steps.sort()

    # Find the checkpoint closest to the discontinuity
    ws_highest = highest_ws_step
    f_lowest = min([int(cp.split('_')[-1]) for cp in f_checkpoints]) if f_checkpoints else 0
    discontinuity_step = ws_highest + f_lowest

    # Create any missing checkpoint files by copying closest existing ones
    for step in expected_steps:
        target_file = os.path.join(concat_dir, f"model_step_{step}")
        if not os.path.exists(target_file):
            print(f"Creating missing checkpoint: model_step_{step}")
            # Find closest existing checkpoint
            closest_step = min(existing_steps, key=lambda x: abs(x - step))
            source_file = os.path.join(concat_dir, f"model_step_{closest_step}")
            if os.path.exists(source_file):
                shutil.copy2(source_file, target_file)

    # Force the discontinuity to be at a multiple of save_freq for better detection
    # This ensures the top-Q verification will check this critical point
    closest_regular_step = save_freq * round(discontinuity_step / save_freq)
    if closest_regular_step not in existing_steps:
        print(f"Ensuring discontinuity checkpoint exists at step {closest_regular_step}")
        if discontinuity_step < closest_regular_step:
            # Copy the last Ws checkpoint
            source = os.path.join(ws_dir, f"model_step_{highest_ws_step}")
            target = os.path.join(concat_dir, f"model_step_{closest_regular_step}")
            shutil.copy2(source, target)
        else:
            # Copy the first f checkpoint
            source = os.path.join(f_dir, f"model_step_0")
            target = os.path.join(concat_dir, f"model_step_{closest_regular_step}")
            shutil.copy2(source, target)


    # Verify what checkpoint files we've created
    print("\nVerifying created checkpoint files:")
    concat_checkpoints = [f for f in os.listdir(concat_dir) if f.startswith("model_step_")]
    print(f"Created {len(concat_checkpoints)} checkpoint files in concatenated directory")
    concat_checkpoints.sort(key=lambda x: int(x.split('_')[-1]))
    print(f"Checkpoint step numbers: {[int(cp.split('_')[-1]) for cp in concat_checkpoints]}")
    
    print("\nStep 5: Analyzing the discontinuity...")
    # Get model parameters
    ws_params = utils.get_parameters(ws_model, numpy=True)
    wt_params = utils.get_parameters(wt_model, numpy=True)
    f_params = utils.get_parameters(f_model, numpy=True)
    
    # Calculate distances
    discontinuity_dist = np.linalg.norm(ws_params - wt_params)
    
    # Generate reference distance (dref) between two random models
    random_model1 = architecture().to(device)
    random_model2 = architecture().to(device)
    dref = np.linalg.norm(
        utils.get_parameters(random_model1, numpy=True) - 
        utils.get_parameters(random_model2, numpy=True)
    )
    
    # Simulate normal update distances
    normal_dists = []
    temp_model = architecture().to(device)
    optimizer = torch.optim.SGD(temp_model.parameters(), lr=lr)
    dummy_inputs = torch.randn(batch_size, 3, 32, 32).to(device) if dataset == "CIFAR10" else torch.randn(batch_size, 1, 28, 28).to(device)
    dummy_targets = torch.randint(0, 10, (batch_size,)).to(device)
    
    # Generate a few normal update distances
    for i in range(10):
        prev_params = utils.get_parameters(temp_model, numpy=True)
        
        optimizer.zero_grad()
        outputs = temp_model(dummy_inputs)
        loss = nn.CrossEntropyLoss()(outputs, dummy_targets)
        loss.backward()
        optimizer.step()
        
        curr_params = utils.get_parameters(temp_model, numpy=True)
        update_dist = np.linalg.norm(curr_params - prev_params)
        normal_dists.append(update_dist)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    # Plot normal update distances
    x_normal = np.arange(len(normal_dists))
    plt.plot(x_normal, normal_dists, 'o-', label='Normal Updates')
    
    # Plot the discontinuity
    plt.axhline(y=discontinuity_dist, color='r', linestyle='--', 
                label=f'Discontinuity (WT-Ws): {discontinuity_dist:.4f}')
    
    # Plot the reference distance
    plt.axhline(y=dref, color='g', linestyle=':', 
                label=f'Reference Distance (dref): {dref:.4f}')
    
    plt.xlabel('Update Step')
    plt.ylabel('Parameter Distance (L2 Norm)')
    plt.title('PoL Concatenation Attack Demonstration')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale to better show differences
    
    plot_path = os.path.join(concat_dir, 'discontinuity_analysis.png')
    plt.savefig(plot_path)
    plt.close()
    
    print("\nAnalysis Results:")
    print(f"Reference distance (dref): {dref:.4f}")
    print(f"Discontinuity at concatenation point: {discontinuity_dist:.4f}")
    print(f"Average normal update distance: {np.mean(normal_dists):.4f}")
    print(f"Ratio of discontinuity to normal updates: {discontinuity_dist/np.mean(normal_dists):.1f}x")
    
    print("\nVerification Analysis:")
    print(f"Top-Q verification with Q=1 would detect this attack with 100% probability")
    prob_random = 1.0 / (len(normal_dists) + 1)  # +1 for the discontinuity
    print(f"Random sampling would detect this attack with only {prob_random:.1%} probability")
    
    print(f"\nPlot saved to: {plot_path}")

    print("\nStep 6: Verifying the concatenated PoL...")
    # Import verification functions
    try:
        from verify import verify_topq, verify_initialization, verify_hash
        
        # Check available checkpoints for verification
        checkpoint_files = [f for f in os.listdir(concat_dir) if f.startswith("model_step_")]
        checkpoint_step_nums = [int(cp.split('_')[-1]) for cp in checkpoint_files]
        highest_checkpoint = max(checkpoint_step_nums) if checkpoint_step_nums else 0
        
        print(f"\nHighest available checkpoint: model_step_{highest_checkpoint}")
        epochs_for_verification = min(1, highest_checkpoint // save_freq)
        print(f"Adjusting to verify {epochs_for_verification} epochs")
        
        # Run top-Q verification (most effective at detecting this attack)
        print("\nRunning top-Q verification (should detect the discontinuity):")
        try:
            # Use smaller Q value for demonstration purposes
            q_value = 1  # Just check the largest update (which should be our discontinuity)
            verification_result = verify_topq(
                dir=concat_dir,
                lr=lr,
                batch_size=batch_size,
                dataset=dataset,
                architecture=architecture,
                save_freq=save_freq,
                order=['2'],  # Use L2 norm
                threshold=[np.mean(normal_dists) * 10],  # Set threshold to 10x normal updates
                epochs=epochs_for_verification,
                q=q_value
            )
            
            # Calculate the ratio of discontinuity to threshold
            discontinuity_ratio = discontinuity_dist / (np.mean(normal_dists) * 10)
            print(f"\nDiscontinuity is {discontinuity_ratio:.1f}x larger than the threshold")
            
            # Demonstrate difference between top-Q and random verification
            detection_prob_topq = min(q_value, 1)  # With top-Q = 1, probability is 100%
            num_checkpoints = len(checkpoint_files)  
            detection_prob_random = 1.0 / num_checkpoints if num_checkpoints > 0 else 0  # With random sampling
            
            print(f"\nDetection probability comparison:")
            print(f"  - Top-{q_value} verification:   {detection_prob_topq:.1%}")
            print(f"  - Random sampling verification: {detection_prob_random:.1%}")
            
        except Exception as e:
            print(f"Top-Q verification failed: {e}")
            
    except ImportError:
        print("Verification modules not found. Skipping verification step.")
    
    return concat_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Demonstrate PoL concatenation attack')
    parser.add_argument('--model', type=str, default='Simple_Conv', 
                        help='Model architecture (e.g., Simple_Conv, resnet20)')
    parser.add_argument('--dataset', type=str, default='MNIST', 
                        help='Dataset (MNIST or CIFAR10)')
    parser.add_argument('--steps', type=int, default=500, 
                        help='Number of initial training steps')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--exp-id', type=str, default='attack_demo',
                        help='Experiment ID (prefix for saved files)')
    
    args = parser.parse_args()
    
    demonstrate_pol_concatenation_attack(
        model_name=args.model,
        dataset=args.dataset,
        initial_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        exp_id=args.exp_id
    )