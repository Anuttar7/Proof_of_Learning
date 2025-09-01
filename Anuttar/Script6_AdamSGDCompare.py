import matplotlib.pyplot as plt
import numpy as np

def lrPlot ():
    x = lr_list

    max_norm_e1 = [[[] for i in range(5)] for j in range(4)]
    max_norm_e2 = [[[] for i in range(5)] for j in range(4)]

    for i in range(5):
        for f in range(len(lr_list)):
            dist = np.array(np.load(f'dists_Adam/{dataset}_{i+1}_b{batch_size}_a{lr_names[f]}.npy'))
            for j in range(4):
                max_norm_e1[j][i].append(np.max(dist[j]) / dref1[j])

    for i in range(5):
        for f in range(len(lr_list)):
            dist = np.array(np.load(f'dists_SGD/{dataset}_{i+1}_b{batch_size}_a{lr_names[f]}.npy'))
            for j in range(4):
                max_norm_e2[j][i].append(np.max(dist[j]) / dref2[j])

    y_mean1 = np.mean(max_norm_e1, axis=1)
    y_min1 = np.min(max_norm_e1, axis=1)
    y_max1 = np.max(max_norm_e1, axis=1)

    y_mean2 = np.mean(max_norm_e2, axis=1)
    y_min2 = np.min(max_norm_e2, axis=1)
    y_max2 = np.max(max_norm_e2, axis=1)

    #L1 Norm
    plt.legend()
    plt.xlabel(f'Learning Rate' + r' $(\eta)$')
    plt.ylabel(r'$max(|| \epsilon _{repr} ||)$')
    plt.plot(x, y_mean1[0], label='Adam')
    plt.fill_between(x, y_min1[0], y_max1[0], alpha=0.3)
    plt.plot(x, y_mean2[0], label='SGD')
    plt.fill_between(x, y_min2[0], y_max2[0], alpha=0.3)
    plt.title(f'{dataset}' + r' $|| W_{t}, W^{\prime}_{t}||_{1}$')
    plt.show()

    #L2 Norm
    plt.legend()
    plt.xlabel(f'Learning Rate' + r' $(\eta)$')
    plt.ylabel(r'$max(|| \epsilon _{repr} ||)$')
    plt.plot(x, y_mean1[1], label='Adam')
    plt.fill_between(x, y_min1[1], y_max1[1], alpha=0.3)
    plt.plot(x, y_mean2[1], label='SGD')
    plt.fill_between(x, y_min2[1], y_max2[1], alpha=0.3)
    plt.title(f'{dataset}' + r' $|| W_{t}, W^{\prime}_{t}||_{2}$')
    plt.show()

    #Infinite Norm
    plt.legend()
    plt.xlabel(f'Learning Rate' + r' $(\eta)$')
    plt.ylabel(r'$max(|| \epsilon _{repr} ||)$')
    plt.plot(x, y_mean1[2], label='Adam')
    plt.fill_between(x, y_min1[2], y_max1[2], alpha=0.3)
    plt.plot(x, y_mean2[2], label='SGD')
    plt.fill_between(x, y_min2[2], y_max2[2], alpha=0.3)
    plt.title(f'{dataset}' + r' $|| W_{t}, W^{\prime}_{t}||_{\infty}$')
    plt.show()

    #Cosine Norm
    plt.legend()
    plt.xlabel(f'Learning Rate' + r' $(\eta)$')
    plt.ylabel(r'$max(|| \epsilon _{repr} ||)$')
    plt.plot(x, y_mean1[3], label='Adam')
    plt.fill_between(x, y_min1[3], y_max1[3], alpha=0.3)
    plt.plot(x, y_mean2[3], label='SGD')
    plt.fill_between(x, y_min2[3], y_max2[3], alpha=0.3)
    plt.title(f'{dataset}' + r' $cos(W_{t}, W^{\prime}_{t})$')
    plt.show()


def StorageOverheadPlot ():
    x = (num_samples / batch_size) / np.array(freq_list)

    max_norm_e1 = [[[] for i in range(5)] for j in range(4)]
    max_norm_e2 = [[[] for i in range(5)] for j in range(4)]

    for i in range(5):
        for f in range(len(freq_list)):
            dist = np.array(np.load(f'dists_Adam/{dataset}_{i+1}_b{batch_size}_k{freq_list[f]}.npy'))
            for j in range(4):
                max_norm_e1[j][i].append(np.max(dist[j]) / dref1[j])

    for i in range(5):
        for f in range(len(freq_list)):
            dist = np.array(np.load(f'dists_SGD/{dataset}_{i+1}_b{batch_size}_k{freq_list[f]}.npy'))
            for j in range(4):
                max_norm_e2[j][i].append(np.max(dist[j]) / dref2[j])


    y_mean1 = np.mean(max_norm_e1, axis=1)
    y_min1 = np.min(max_norm_e1, axis=1)
    y_max1 = np.max(max_norm_e1, axis=1)

    y_mean2 = np.mean(max_norm_e2, axis=1)
    y_min2 = np.min(max_norm_e2, axis=1)
    y_max2 = np.max(max_norm_e2, axis=1)

    #L1 Norm
    plt.legend()
    plt.xlabel("Storage Overhead (S/k)")
    plt.xticks([i for i in range(0, 90, 10)], [str(i) + 'x' for i in range(0, 90, 10)])
    plt.ylabel(r'$max(|| \epsilon _{repr} ||)$')
    plt.plot(x, y_mean1[0], label='Adam')
    plt.fill_between(x, y_min1[0], y_max1[0], alpha=0.3)
    plt.plot(x, y_mean2[0], label='SGD')
    plt.fill_between(x, y_min2[0], y_max2[0], alpha=0.3)
    plt.title(f'{dataset}' + r' $|| W_{t}, W^{\prime}_{t}||_{1}$')
    plt.show()

    #L2 Norm
    plt.legend()
    plt.xlabel("Storage Overhead (S/k)")
    plt.xticks([i for i in range(0, 90, 10)], [str(i) + 'x' for i in range(0, 90, 10)])
    plt.ylabel(r'$max(|| \epsilon _{repr} ||)$')
    plt.plot(x, y_mean1[1], label='Adam')
    plt.fill_between(x, y_min1[1], y_max1[1], alpha=0.3)
    plt.plot(x, y_mean2[1], label='SGD')
    plt.fill_between(x, y_min2[1], y_max2[1], alpha=0.3)
    plt.title(f'{dataset}' + r' $|| W_{t}, W^{\prime}_{t}||_{2}$')
    plt.show()

    #Infinite Norm
    plt.legend()
    plt.xlabel("Storage Overhead (S/k)")
    plt.xticks([i for i in range(0, 90, 10)], [str(i) + 'x' for i in range(0, 90, 10)])
    plt.ylabel(r'$max(|| \epsilon _{repr} ||)$')
    plt.plot(x, y_mean1[2], label='Adam')
    plt.fill_between(x, y_min1[2], y_max1[2], alpha=0.3)
    plt.plot(x, y_mean2[2], label='SGD')
    plt.fill_between(x, y_min2[2], y_max2[2], alpha=0.3)
    plt.title(f'{dataset}' + r' $|| W_{t}, W^{\prime}_{t}||_{\infty}$')
    plt.show()

    #Cosine Norm
    plt.legend()
    plt.xlabel("Storage Overhead (S/k)")
    plt.xticks([i for i in range(0, 90, 10)], [str(i) + 'x' for i in range(0, 90, 10)])
    plt.ylabel(r'$max(|| \epsilon _{repr} ||)$')
    plt.plot(x, y_mean1[3], label='Adam')
    plt.fill_between(x, y_min1[3], y_max1[3], alpha=0.3)
    plt.plot(x, y_mean2[3], label='SGD')
    plt.fill_between(x, y_min2[3], y_max2[3], alpha=0.3)
    plt.title(f'{dataset}' + r' $cos(W_{t}, W^{\prime}_{t})$')
    plt.show()



def CheckpointIntervalPlot ():
    x = freq_list

    max_norm_e1 = [[[] for i in range(5)] for j in range(4)]
    max_norm_e2 = [[[] for i in range(5)] for j in range(4)]

    for i in range(5):
        for f in range(len(freq_list)):
            dist = np.array(np.load(f'dists_Adam/{dataset}_{i+1}_b{batch_size}_k{freq_list[f]}.npy'))
            for j in range(4):
                max_norm_e1[j][i].append(np.max(dist[j]) / dref1[j])

    for i in range(5):
        for f in range(len(freq_list)):
            dist = np.array(np.load(f'dists_SGD/{dataset}_{i+1}_b{batch_size}_k{freq_list[f]}.npy'))
            for j in range(4):
                max_norm_e2[j][i].append(np.max(dist[j]) / dref2[j])


    y_mean1 = np.mean(max_norm_e1, axis=1)
    y_min1 = np.min(max_norm_e1, axis=1)
    y_max1 = np.max(max_norm_e1, axis=1)

    y_mean2 = np.mean(max_norm_e2, axis=1)
    y_min2 = np.min(max_norm_e2, axis=1)
    y_max2 = np.max(max_norm_e2, axis=1)

    #L1 Norm
    plt.legend()
    plt.xlabel("Checkpoint Interval (k)")
    plt.ylabel(r'$max(|| \epsilon _{repr} ||)$')
    plt.plot(x, y_mean1[0], label='Adam')
    plt.fill_between(x, y_min1[0], y_max1[0], alpha=0.3)
    plt.plot(x, y_mean2[0], label='SGD')
    plt.fill_between(x, y_min2[0], y_max2[0], alpha=0.3)
    plt.title(f'{dataset}' + r' $|| W_{t}, W^{\prime}_{t}||_{1}$')
    plt.show()

    #L2 Norm
    plt.legend()
    plt.xlabel("Checkpoint Interval (k)")
    plt.ylabel(r'$max(|| \epsilon _{repr} ||)$')
    plt.plot(x, y_mean1[1], label='Adam')
    plt.fill_between(x, y_min1[1], y_max1[1], alpha=0.3)
    plt.plot(x, y_mean2[1], label='SGD')
    plt.fill_between(x, y_min2[1], y_max2[1], alpha=0.3)
    plt.title(f'{dataset}' + r' $|| W_{t}, W^{\prime}_{t}||_{2}$')
    plt.show()

    #Infinite Norm
    plt.legend()
    plt.xlabel("Checkpoint Interval (k)")
    plt.ylabel(r'$max(|| \epsilon _{repr} ||)$')
    plt.plot(x, y_mean1[2], label='Adam')
    plt.fill_between(x, y_min1[2], y_max1[2], alpha=0.3)
    plt.plot(x, y_mean2[2], label='SGD')
    plt.fill_between(x, y_min2[2], y_max2[2], alpha=0.3)
    plt.title(f'{dataset}' + r' $|| W_{t}, W^{\prime}_{t}||_{\infty}$')
    plt.show()

    #Cosine Norm
    plt.legend()
    plt.xlabel("Checkpoint Interval (k)")
    plt.ylabel(r'$max(|| \epsilon _{repr} ||)$')
    plt.plot(x, y_mean1[3], label='Adam')
    plt.fill_between(x, y_min1[3], y_max1[3], alpha=0.3)
    plt.plot(x, y_mean2[3], label='SGD')
    plt.fill_between(x, y_min2[3], y_max2[3], alpha=0.3)
    plt.title(f'{dataset}' + r' $cos(W_{t}, W^{\prime}_{t})$')
    plt.show()


if __name__ == '__main__':
    epochs = 10
    batch_size = 128
    dataset = "OrganAMNIST"
    num_samples = 10000

    dref1 = np.array(np.load(f'dref/{dataset}_Adam_dref.npy'))
    dref2 = np.array(np.load(f'dref/{dataset}_SGD_dref.npy'))

    freq_list = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400]
    lr_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    lr_names = ['1e-3', '5e-3', '1e-2', '5e-2', '1e-1', '5e-1', '1e0']
    
    CheckpointIntervalPlot()
    StorageOverheadPlot()
    lrPlot()