from medmnist import OrganAMNIST
import numpy as np

train_set = OrganAMNIST(split='train', download=True)

x = train_set.imgs
y = train_set.labels

padded_x = np.pad(x, pad_width=((0,0), (2,2), (2,2)), mode='constant', constant_values=0)

print(type(padded_x))
print(padded_x.shape)