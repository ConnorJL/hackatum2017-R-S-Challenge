import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from capsule_layer import CapsuleLayer

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1)

        s:elf.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=128, out_channels=32, kernel_size=3, stride=2)
        self.category_capsules = CapsuleLayer(num_capsules=59, num_route_nodes=1000, in_channels = 32, out_channels=16)


    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        #cats = self.category_capsules(x)

        #cats = (cats ** 2).sum(dim=-1) ** 0.5
        #cats = F.softmax(cats)
        print(x.size())
        return x



class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()

    def forward(self, labels, classes):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        return margin_loss


preprocess = transforms.Compose([
    transforms.ToTensor()
])
