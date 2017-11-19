import pickle as pkl
import torch
import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class RSDataset(Dataset):
    def __init__(self, path, grey=False, transform=None, train=True, size=(180, 100)):
        with open(os.path.join(path, "master.pkl"), "rb") as f:
            if train:
                self.elements = pkl.load(f)["train"]
            else:
                self.elements = pkl.load(f)["test"]

        for e in self.elements:
            e.img_path = os.path.join(path, e.img_path)
        with open(os.path.join(path, "labels.pkl"), "rb") as f:
            self.labels = pkl.load(f)
        self.transform = transform
        self.grey = grey
        self.size = size

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, idx):
        img_name = self.elements[idx].img_path
        image = Image.open(img_name)
        if self.grey:
            image = image.convert('L')
        image = image.resize(self.size)

        if self.transform:
            image = self.transform(image)
        labels = np.zeros(59)
        if not len(self.elements[idx].labels) == 0:
            labels[self.labels[self.elements[idx].labels[0].name]] = 1
        labels = torch.FloatTensor(labels)
        sample = {"image": image, "labels": labels}

        return sample
