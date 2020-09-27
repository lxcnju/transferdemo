import os
import copy
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms

from collections import Counter

from utils import load_pickle

from paths import data_dir


def load_data(domain):
    """ load data from pickle
    """
    fpath = os.path.join(data_dir, "{}_demo.pkl".format(domain))
    obj = load_pickle(fpath)
    xs, ys = obj["images"], obj["labels"]
    print(domain, xs.shape, ys.shape, Counter(ys))
    return xs, ys


class DigitsDataset(data.Dataset):
    def __init__(self, images, labels, is_train=True):
        self.images = copy.deepcopy(images)
        self.labels = copy.deepcopy(labels)
        self.is_train = is_train

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]

        # transforms.ToPILImage need (H, W, C) np.uint8 input
        img = img.transpose(1, 2, 0).astype(np.uint8)

        # return (C, H, W) tensor
        img = self.transform(img)

        label = torch.LongTensor([label])[0]
        return img, label
