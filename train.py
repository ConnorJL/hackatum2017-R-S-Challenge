import os
import pickle as pkl
from random import shuffle

import numpy as np
import torch
from torch.utils.data import Dataloader

from dataset import RSDataset

batch_size = 16
num_epochs = 100
net = "politically_correct"
master_list_location = "/home/connor/Projects/hackatum2017"

if net == "politically_correct":
    from politically_import import network, preprocess, loss

if not os.path.exists(net):
    os.mkdir(net)
    restore = False
else:
    # restore = attempt_to_restore()
    pass

if __name__ == "__main__":
    model = network()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = loss()
    train_data = RSDataset(master_list_location, grey=True, transform=preprocess)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    print("Number of parameters: ", sum(param.numel() for param in model.parameters()))

    if restore:
        model.load_state_dict(statestuff)
        optimizier.restore_optim

    model.cuda()

    for i in range(num_epochs):
        for e in train_dataloader:
            model.zero_grads()

            out = model(e["image"])
            loss = criterion(out, e["labels"])

            loss.backward()
            optmizer.step()
