import os
import pickle as pkl
from random import shuffle

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RSDataset

batch_size = 16
num_epochs = 100
net = "politically_correct"
master_list_location = "/home/connor1995/train"

if net == "politically_correct":
    from politically_correct import network, preprocess, loss

def save_checkpoint(state, filename='checkpoint'):
    filename = os.path.join(net, filename+ "_" + str(state["epoch"]) + ".pth.tar")
    torch.save(state, filename)


if not os.path.exists(net):
    os.mkdir(net)
    restore = False
else:
    restore = True
    l = []
    for f in os.listdir(net):
        l.append(int(f.split(".")[0].split("_")[1]))
    l.sort()
    resume_path = os.path.join(net, "checkpoint" + "_"+ str(l[-1]) + ".pth.tar")

debug = True
if __name__ == "__main__":
    model = network()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = loss()
    train_data = RSDataset(master_list_location, grey=True, transform=preprocess)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    print("Number of parameters: ", sum(param.numel() for param in model.parameters()))
    start_epoch = 0
    if restore:
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
    model.cuda()

    for i in range(start_epoch, num_epochs):
        for e in train_dataloader:
            model.zero_grad()
            if debug:
                print(e["image"].size())

            img = e["image"]
            out = model(img)
            loss = criterion(e["labels"], out)

            loss.backward()
            optimizer.step()

            save_checkpoint({
            'epoch': epoch + 1,
            'arch': net,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            })

