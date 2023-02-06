import argparse
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from functools import partial
import albumentations as A
from albumentations.pytorch.transforms import ToTensor

from utils import *
from models import s6_cifar10

cv = partial(nn.Conv2d, bias=False)
bn = nn.BatchNorm2d
relu = nn.ReLU

# optimize operations if available (https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/24)
torch.backends.cudnn.benchmark = True

BATCH = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataloader_args = dict(shuffle=True, batch_size=BATCH, num_workers=2, pin_memory=True)\
    if torch.cuda.is_available() else dict(shuffle=True, batch_size=32)


arg_parser = argparse.ArgumentParser(
    description="Training program for various models with multiple options.")

arg_parser.add_argument('--lr', default=0.01, type=float,
                        help="Learning rate to set for the model")

arg_parser.add_argument('--dataset', default=None,
                        type=str, help="The dataset to use.")

arg_parser.add_argument('--model', default=None, type=str,
                        help="The model to use for training.")

args = arg_parser.parse_args('--epochs', default=1, type=int,
                             help="Number of epochs to run the training for.")


if args.dataset.lower() == "cifar10":
    mean, sdev = get_mean_and_std(torchvision.datasets.CIFAR10(root="./data",
                                                               train=True,
                                                               download=True,
                                                               transform=transforms.Compose([transforms.ToTensor()])))

    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15,
                           rotate_limit=30, p=0.20),
        A.CoarseDropout(max_holes=1, p=0.15, max_height=16,
                        max_width=16, min_holes=1, min_height=16,
                        min_width=16, fill_value=mean),
        # A.MedianBlur(blur_limit=3, p=0.1),
        A.HueSaturationValue(p=0.1),
        #   A.GaussianBlur(blur_limit=3, p=0.12),
        # A.RandomBrightnessContrast(brightness_limit=0.09,contrast_limit=0.1, p=0.15),
        A.Normalize(mean=mean, std=sdev),
        ToTensor()
    ])

    test_transforms = A.Compose([
                                A.Normalize(mean=mean, std=sdev),
                                ToTensor()
                                ])


train_losses = []
train_accuracy = []
test_losses = []
test_accuracy = []


def train_eval_model(model, train_loader, optimizer, device, epochs=1, test=False, test_loader=None, scheduler=None):

    model.train()  # set the train mode

    # iterate over for `epochs` epochs and keep storing valuable info

    for epoch in range(epochs):
        correct = processed = train_loss = 0

        print(f"\n epoch num ================================= {epoch+1}")

        pbar = tqdm(train_loader)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(
                device)  # move data to `device`

            optimizer.zero_grad()  # zero out the gradients to avoid accumulating them over loops

            output = model(data)  # get the model's predictions

            # calculate Negative Log Likelihood loss using ground truth labels and the model's predictions
            loss = F.nll_loss(output, target)

            train_loss += loss.item()  # add up the train loss

            loss.backward()  # The magic function to perform backpropagation and calculate the gradients

            optimizer.step()  # take 1 step for the optimizer and update the weights

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            # compare and see how many predictions are coorect and then add up the count
            correct += pred.eq(target.view_as(pred)).sum().item()

            processed += len(data)  # total processed data size

        acc = 100 * correct/processed

        train_losses.append(train_loss)

        train_accuracy.append(acc)

        if scheduler:
            print("\n\n\t\t\tLast LR -->", scheduler.get_last_lr())
            scheduler.step()

        pbar.set_description(desc=f'loss={loss.item()} batch_id={batch_idx}')

        train_loss /= len(train_loader.dataset)

        print('\n\t\t\tTrain metrics: accuracy: {}/{} ({:.4f}%)'.format(correct,
                                                                        len(train_loader.dataset),
                                                                        correct * 100 / len(train_loader.dataset)))

        if test:  # moving to evaluation
            model.eval()  # set the correct mode

            correct = test_loss = 0

            with torch.no_grad():  # to disable gradient calculation with no_grad context

                for data, target in test_loader:

                    data, target = data.to(device), target.to(device)

                    output = model(data)

                    # sum up batch loss
                    test_loss += F.nll_loss(output,
                                            target, reduction='sum').item()

                    # get the index of the max log-probability
                    pred = output.argmax(dim=1, keepdim=True)

                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            test_accuracy.append(100. * correct / len(test_loader.dataset))

            print('\n\tTest metrics: average loss: {:.4f}, accuracy: {}/{} ({:.5f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))


train_set = Cifar10SearchDataset(
    train=True, download=True, transform=train_transforms)

test_set = Cifar10SearchDataset(
    train=False, download=True, transform=test_transforms)


# data loaders on data sets
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, **dataloader_args)

test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)

model = s6_cifar10()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
step_lr = StepLR(optimizer=optimizer, step_size=2, gamma=0.81)


train_eval_model(model, train_loader, optimizer, device,
                 test=True,
                 test_loader=test_loader,
                 scheduler=step_lr,
                 epochs=args.epochs)
