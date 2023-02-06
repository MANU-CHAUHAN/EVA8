import os
import sys
import time
import math
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torch.optim as optim


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean.tolist(), std.tolist()


class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="./data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def plot_graphs(*, train_losses, train_accuracy, test_losses, test_accuracy):
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_accuracy)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_accuracy)
    axs[1, 1].set_title("Test Accuracy")


def get_optmimizer(model, optim_type="SGD", lr=0.001, enable_nesterov=False, momentum_value=0.9):
    if optim_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=momentum_value, nesterov=enable_nesterov)
        return optimizer
    # step_lr = StepLR(optimizer=optimizer, step_size=1, gamma=0.87)


if __name__ == "__main__":
    mean, sdev = get_mean_and_std(torchvision.datasets.CIFAR10(root="./data", train=True,
                                                               download=True, transform=transforms.Compose([transforms.ToTensor()])))

    print(mean, sdev)
