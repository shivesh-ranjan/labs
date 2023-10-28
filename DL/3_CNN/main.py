import torch
import torch.nn as nn
from torch.nn.modules import padding
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

torch.manual_seed(42)

data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

import os

data_dir = ""

image_datasets = {
        x: datasets.ImageFolder(
            root=os.path.join(data_dir, x),
            transform=data_transforms[x]
            )for x in ['train', 'val']
        }

dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=32,
            shuffle=True,
            num_workers=4
            )for x in ['train', 'val']
        }

class_names = image_datasets['train'].classes
print("Classes in the datasets: \n", class_names)

for x in ['train', 'val']:
    print(f"{x} dataset len: ", len(image_datasets[x]))
    print(f"{x} dataloader len: ", len(dataloaders[x]))





checkpoint_dir = "checkpoints"

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7*7*32, 600)
