import argparse
import os
import hashlib
import numpy as np
import torch
import torchvision
from functools import reduce

import utils
from train import train
import model as custom_model
import matplotlib.pyplot as plt

def calc_dist(dir, lr, batch_size, dataset, architecture, save_freq, order, threshold, half=0):
    if not os.path.exists(dir):
        raise FileNotFoundError("Model directory not found")
    sequence = np.load(os.path.join(dir, "indices.npy"))

    if not isinstance(order, list):
        order = [order]
        threshold = [threshold]
    else:
        assert len(order) == len(threshold)

    dist_list = [[] for i in range(len(order))]

    target_model = os.path.join(dir, f"model_step_0")
    for i in range(0, sequence.shape[0], save_freq):
        previous_state = target_model
        if i + save_freq >= sequence.shape[0]:
            target_model = os.path.join(dir, f"model_step_{sequence.shape[0]}")
            reproduce = train(lr, batch_size, 0, dataset, architecture, model_dir=previous_state,
                              sequence=sequence[i:], half=half)
        else:
            target_model = os.path.join(dir, f"model_step_{i + save_freq}")
            reproduce = train(lr, batch_size, 0, dataset, architecture, model_dir=previous_state,
                              sequence=sequence[i:i+save_freq], half=half)
        res = utils.parameter_distance(target_model, reproduce, order=order,
                                       architecture=architecture, half=half)
        for j in range(len(order)):
            dist_list[j].append(res[j])

    dist_list = np.array(dist_list)
    return dist_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="OrganAMNIST")
    parser.add_argument('--model', type=str, default="resnet20",
                        help="models defined in model.py or any torchvision model.\n"
                             "Recommendation for CIFAR-10: resnet20/32/44/56/110/1202\n"
                             "Recommendation for CIFAR-100: resnet18/34/50/101/152"
                        )
    parser.add_argument('--model-dir', help='path/to/the/proof', type=str, default='proof_SGD')
    parser.add_argument('--save-freq', type=int, default=100, help='frequence of saving checkpoints')
    parser.add_argument('--dist', type=str, nargs='+', default=['1', '2', 'inf', 'cos'],
                        help='metric for computing distance, cos, 1, 2, or inf')
    parser.add_argument('--q', type=int, default=2, help="Set to >1 to enable top-q verification,"
                                                         "otherwise all steps will be verified.")
    parser.add_argument('--delta', type=float, default=[1000, 10, 0.1, 0.01],
                        help='threshold for verification')

    arg = parser.parse_args()

    try:
        architecture = eval(f"custom_model.{arg.model}")
    except:
        architecture = eval(f"torchvision.models.{arg.model}")

    dref = np.array(np.load(f'dref/{arg.dataset}_SGD_dref.npy'))

    freq_list = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400]

    max_norm_e = [[[] for j in range(5)] for i in range(len(arg.dist))]

    for f in range(len(freq_list)):
        for i in range(5):
            model_dir = os.path.join(arg.model_dir, f"{arg.dataset}_{i+1}_b{arg.batch_size}_k{freq_list[f]}")
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Model directory {model_dir} not found")
            
            # Verify the model
            print(f"Verifying model with save frequency {freq_list[f]}...")
            dists = calc_dist(model_dir, arg.lr, arg.batch_size, arg.dataset, architecture, freq_list[f],
                            arg.dist, arg.delta)
            
            dists = np.array(dists)
            save_path_dist = f'dists_SGD/{arg.dataset}_{i+1}_b{arg.batch_size}_k{freq_list[f]}.npy'
            np.save(save_path_dist, dists)

            # Calculate the maximum distance
            for j in range(len(arg.dist)):
                max_norm_e[j][i].append(np.max(dists[j]) / dref[j])

    # print(max_norm_e)

    x = freq_list
    y_mean = np.mean(max_norm_e, axis=1)
    y_min = np.min(max_norm_e, axis=1)
    y_max = np.max(max_norm_e, axis=1)
    plt.plot(x, y_mean[0], label=r'$|| W_{t}, W^{\prime}_{t}||_{1}$')
    plt.fill_between(x, y_min[0], y_max[0], alpha=0.3)
    plt.plot(x, y_mean[1], label=r'$|| W_{t}, W^{\prime}_{t}||_{2}$')
    plt.fill_between(x, y_min[1], y_max[1], alpha=0.3)
    plt.plot(x, y_mean[2], label=r'$|| W_{t}, W^{\prime}_{t}||_{\infty}$')
    plt.fill_between(x, y_min[2], y_max[2], alpha=0.3)
    plt.plot(x, y_mean[3], label=r'$cos(W_{t}, W^{\prime}_{t})$')
    plt.fill_between(x, y_min[3], y_max[3], alpha=0.3)
    
    plt.xlabel("Checkpoint Interval (k)")
    plt.ylabel(r'$max(|| \epsilon _{repr} ||)$')
    plt.title("OrganAMNIST SGD")
    plt.legend()
    plt.show()