import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Define a simplified training function using the Adam optimizer
def train_with_adam(model, dataset_name, batch_size=128, lr=0.001, epochs=10):
    """
    Train a neural network using the Adam optimizer.

    Args:
        model: PyTorch model to train.
        dataset_name: Name of the dataset (e.g., 'CIFAR10', 'MNIST').
        batch_size: Batch size for training.
        lr: Learning rate for the Adam optimizer.
        epochs: Number of training epochs.
    """
    # Set device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load dataset
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset. Use 'CIFAR10' or 'MNIST'.")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get inputs and labels, and move them to the device
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("Finished Training")

# Example usage (incomplete, requires a model to be defined)
# from model import Simple_Conv
# model = Simple_Conv()
# train_with_adam(model, dataset_name='CIFAR10', batch_size=128, lr=0.001, epochs=10)