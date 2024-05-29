from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from torch.utils.data.sampler import RandomSampler
from itertools import repeat
import pandas as pd
import numpy as np
import torch
from king_housing_preprocessing import *
import pdb
import tqdm
from scipy.stats import truncnorm


class CIFARDataset(Dataset):
    def __init__(self, lst):
        self.lst = lst

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        return self.lst[idx]

class CBDataset(Dataset):
    def __init__(self, context, label, loc=None):
        self.context = context
        self.label = label
        self.loc = loc

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        context_val = torch.tensor(self.context.loc[idx].values, dtype=torch.float)
        label_val = torch.tensor(self.label.loc[idx]).to(torch.float)
        if self.loc is not None:
            loc_val = torch.tensor(self.loc[idx]).squeeze().to(torch.float)
            return context_val, label_val, loc_val
        else:
            return context_val, label_val


# convert dataloader into infinite iterator
def infinite_dataloader(dataloader):
    for _ in repeat(dataloader):
        yield from dataloader


def process_cifar100(cfg, accelerator):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4866, 0.4409], [0.2673, 0.2564, 0.2762]),
        ]
    )

    # fill in with path to desired location of CIFAR data
    cifar100_train_data = torchvision.datasets.CIFAR100(
        root="datasets/cifar", train=True, transform=train_transform, download=True
    )

    train_dataloader = DataLoader(
        cifar100_train_data,
        batch_size=cfg.task.batch_size,
        shuffle=True,
        drop_last=True,
    )

    num_actions = 100

    dataloader = accelerator.prepare(train_dataloader)

    return infinite_dataloader(dataloader), num_actions



def process_prudential(cfg, accelerator):
    original_data = pd.read_csv(cfg.task.prudential.original)
    data = pd.read_csv(cfg.task.prudential.processed)

    num_actions = 8
    dataset = CBDataset(data, original_data["Response"])
    # pdb.set_trace()
    train_dataloader = DataLoader(
        dataset, batch_size=cfg.task.batch_size, drop_last=True, shuffle=True
    )
    train_dataloader = accelerator.prepare(train_dataloader)

    return infinite_dataloader(train_dataloader), num_actions


def process_housing(cfg, accelerator):
    num_actions = 100

    dataset = ArffToPytorch(
        cfg.task.arff_file,
        target="price",
        skipcol=["id"],
        skiprow=lambda z: z["price"] > 1e6,
    )

    train_dataloader = DataLoader(
        dataset, batch_size=cfg.task.batch_size, drop_last=True, shuffle=True
    )

    train_dataloader = accelerator.prepare(train_dataloader)

    return infinite_dataloader(train_dataloader), num_actions


def get_dataloader(cfg, accel):
    if cfg.task.task == "cifar100":
        return process_cifar100(cfg, accel)
    elif cfg.task.task == "prudential":
        return process_prudential(cfg, accel)
    elif cfg.task.task == "housing":
        return process_housing(cfg, accel)
    else:
        raise NotImplementedError(
            f"Task {cfg.task} not implemented. Please choose on "
        )
