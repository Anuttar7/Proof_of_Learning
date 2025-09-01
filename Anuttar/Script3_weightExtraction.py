import argparse
import model as custom_model
import torch
import numpy as np
from train import train
from utils import parameter_distance
import os

def extract_weights(trained_model):
    """
    Extract and flatten weights from a trained model into a single list
    Args:
        trained_model: The trained PyTorch model
    Returns:
        list: Flattened weights from the model
    """
    # Handle DataParallel wrapper if present
    if isinstance(trained_model, torch.nn.DataParallel):
        trained_model = trained_model.module
        
    # Extract all parameters and convert to numpy arrays
    weights = []
    for param in trained_model.parameters():
        weights.append(param.data.cpu().numpy().flatten())
        
    # Concatenate all flattened arrays
    all_weights = np.concatenate(weights)
    print(f"Number of weights extracted: {len(all_weights)}")
    
    return all_weights



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dataset', type=str, default="OrganAMNIST")
    parser.add_argument('--model', type=str, default="resnet20",
                        help="models defined in model.py or any torchvision model.\n"
                             "Recommendation for CIFAR-10: resnet20/32/44/56/110/1202\n"
                             "Recommendation for CIFAR-100: resnet18/34/50/101/152"
                        )
    parser.add_argument('--id', help='experiment id', type=str, default='XX')
    parser.add_argument('--save-freq', type=int, default=100, help='frequence of saving checkpoints')
    parser.add_argument('--num-gpu', type=int, default=torch.cuda.device_count())
    parser.add_argument('--milestone', nargs='+', type=int, default=[100, 150])
    parser.add_argument('--verify', type=int, default=0)
    arg = parser.parse_args()

    print(f'trying to allocate {arg.num_gpu} gpus')
    try:
        architecture = eval(f"custom_model.{arg.model}")
    except:
        architecture = eval(f"torchvision.models.{arg.model}")
    
    trained_model1 = train(arg.lr, 128, arg.epochs, arg.dataset, architecture, exp_id=str(arg.id + str(1)),
                          save_freq=arg.save_freq, num_gpu=arg.num_gpu, dec_lr=arg.milestone,
                          verify=arg.verify, resume=True)
    trained_model2 = train(arg.lr, 256, arg.epochs, arg.dataset, architecture, exp_id=str(arg.id + str(2)),
                          save_freq=arg.save_freq, num_gpu=arg.num_gpu, dec_lr=arg.milestone,
                          verify=arg.verify, resume=True)
    
    # Extract and save weights for model1
    weights1 = extract_weights(trained_model1)
    
    # Extract and save weights for model2
    weights2 = extract_weights(trained_model2)

    res = parameter_distance(weights1, weights2, order=['1', '2', 'inf', 'cos'], architecture=architecture)
    
    save_path_dref = f'dref/{arg.dataset}_SGD_dref.npy'
    np.save(save_path_dref, res)

    print(len(weights1))
    print(len(weights2))
    print("____")
    print(res)

    res_load = np.load(f'dref/OrganAMNIST_SGD_dref.npy')
    print(res_load)