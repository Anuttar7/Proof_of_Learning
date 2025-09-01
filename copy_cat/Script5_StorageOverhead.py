import matplotlib.pyplot as plt
import numpy as np

dref = np.array(np.load('dref/MNIST_dref.npy'))
freq_list = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400]
max_norm_e = [[[] for i in range(5)] for j in range(4)]

epochs = 10
batch_size = 128
dataset = "MNIST"
num_samples = 10000

for i in range(5):
    for f in range(len(freq_list)):
        dist = np.array(np.load(f'dists/{dataset}_{i+1}_b{batch_size}_k{freq_list[f]}.npy'))
        for j in range(4):
            max_norm_e[j][i].append(np.max(dist[j]) / dref[j])

y_mean = np.mean(max_norm_e, axis=1)
y_min = np.min(max_norm_e, axis=1)
y_max = np.max(max_norm_e, axis=1)

x = (num_samples / batch_size) / np.array(freq_list)

plt.plot(x, y_mean[0], label=r'$|| W_{t}, W^{\prime}_{t}||_{1}$')
plt.fill_between(x, y_min[0], y_max[0], alpha=0.3)
plt.plot(x, y_mean[1], label=r'$|| W_{t}, W^{\prime}_{t}||_{2}$')
plt.fill_between(x, y_min[1], y_max[1], alpha=0.3)
plt.plot(x, y_mean[2], label=r'$|| W_{t}, W^{\prime}_{t}||_{\infty}$')
plt.fill_between(x, y_min[2], y_max[2], alpha=0.3)
plt.plot(x, y_mean[3], label=r'$cos(W_{t}, W^{\prime}_{t})$')
plt.fill_between(x, y_min[3], y_max[3], alpha=0.3)

plt.legend()
plt.xlabel("Storage Overhead (S/k)")
plt.xticks([i for i in range(0, 90, 10)], [str(i) + 'x' for i in range(0, 90, 10)])
plt.ylabel(r'$max(|| \epsilon _{repr} ||)$')
plt.title("MNIST Returns")
plt.show()