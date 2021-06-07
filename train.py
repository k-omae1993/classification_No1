from __future__ import print_function, division

import torch
from torch._C import parse_schema
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from model import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for Classification")
    parser.add_argument("--num_epochs", type=int, default=1, help="path to num_epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="path to lr")
    parser.add_argument("--save_file", type=str, default="best_model", help="path to save_file")
    args = parser.parse_args()


    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9)
    exp_lr_schefuler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_schefuler, num_epochs=args.num_epochs)

    torch.save(model_ft.state_dict(), 'save_models/'+args.save_file)