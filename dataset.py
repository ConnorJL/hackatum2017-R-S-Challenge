import pickle as pkl
import torch
import os

from skimage.io import imread
from torch.utils.data import Dataset


class RSDataset(Dataset):
    def __init__(self, path, grey=False, transform=None, train=True):
        with open(os.path.join(path, "master.pkl"), "rb") as f:
            if train:
                l = pkl.load(f)["train"]
            else:
                l = pkl.load(f)["test"]

        self.elements = []
        for i in l:
            self.elements.append(os.path.join(path, i))

        with open(os.path.join(path, "labels.pkl", "rb")) as f:
            self.labels = pkl.load(f)
        self.transform = transform
        self.grey = grey

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, idx):
        img_name = self.elements[idx].img_path
        image = imread(img_name,as_grey=self.grey)

        if self.transform:
            image = transform(image)
        labels = self.labels[self.elements[idx].labels[0]]
        sample = {"image": image, "labels": labels}
