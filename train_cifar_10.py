import os
import argparse
import time
import shutil
import matplotlib.pyplot as plt
import argparse

import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
import torchvision.datasets as datasets
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from data import data
from data import utils
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from models.classifiers.darknet.darknet53 import Darknet53
from models.classifiers.darknet.darknet53_msa import Darknet53 as MSADarknet53

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1.25e-4)
    parser.add_argument('--batchsize', type=int, default=96) # for cifar
    parser.add_argument('--valbatchsize', type=int, default=1024)
    parser.add_argument('--model', type=str, default='darknet53')
    return parser.parse_args()

# Do Xaiver-based weight initalization
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias:
            torch.nn.init.xavier_uniform_(m.bias)

def val(model, loader, classes):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    model.eval()

    # again no gradients needed
    with torch.no_grad():
        for images, labels in loader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    average_accuracy = 0.0
    for classname, correct_count in correct_pred.items():
        accuracy = float(correct_count) / total_pred[classname]
        average_accuracy += accuracy
    
    average_accuracy = average_accuracy/len(correct_pred)    
    model.train()
    return average_accuracy

def save_train_model(model, optimizer, epoch, lr_sched, outPath):
    save_dict = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer,
        "epoch": epoch,
        "lr_sched": lr_sched
    }
    torch.save(save_dict, outPath)

def load_train_model(inPath, model, optimizer, epoch, lr_sched):
    loaded_data = torch.load(inPath)
    model.load_state_dict(loaded_data['state_dict'])
    optimizer = loaded_data['optimizer']
    epoch = loaded_data['epoch']
    lr_sched = loaded_data['lr_sched']
    return optimizer, epoch, lr_sched


def train(model, opt):
    # hyperparameters:
    EPOCHS = opt.epochs
    LR = opt.lr
    MOMENTUM = 0.9
    BATCH_SIZE=opt.batchsize
    VAL_BATCH_SIZE=opt.valbatchsize

    # preprocess for training and testing loaders
    metadata = data.CIFAR10
    preprocess_train = metadata.train_transform()
    preprocess_test = metadata.test_transform()

    # make CIFAR loader:
    cifar10_train = datasets.CIFAR10('./data', train=True, download=True, transform=preprocess_train)
    cifar10_test = datasets.CIFAR10('./data', train=False, download=True, transform=preprocess_test)

    trainloader = DataLoader(cifar10_train, batch_size=BATCH_SIZE) 
    testloader = DataLoader(cifar10_test, batch_size=VAL_BATCH_SIZE)

    # training loop:
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-2 )
    loss = nn.CrossEntropyLoss().cuda()

    # replace loss function with label smoothing
    model = model.cuda()
    cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=cosine_annealing_scheduler)
    
    # plot scheduler:
    '''
    _test = []
    for epoch in range(350):
        _test.append(scheduler.get_lr()[0])
        scheduler.step(epoch)
    print(_test)
    np.save('./graph_scripts/lr_dk53.npy', np.array(_test))
    exit()
    '''
    
    # create validation folder to save weights:
    best_val_score = 0.0
    if opt.exp == '':
        exp_folder = utils.create_exp_folder('runs')
    else: 
        exp_folder = opt.exp
    print("Writing to runs/", exp_folder)
    weight_file = os.path.join("./runs/%s"%exp_folder, "best_darknet53_model.pth")
    writer = SummaryWriter('./runs/%s'%(exp_folder))

    # Main training loop:
    for epoch in tqdm(range(EPOCHS), desc="Epoch"):
        train_loss = 0
        correct = 0
        total = 0
        
        print("current optimizer lr: %f"%(optimizer.param_groups[0]['lr']))

        for img, gt_label in tqdm(trainloader, desc="Training"):
            img = img.cuda()
            gt_label = gt_label.cuda()
            optimizer.zero_grad()
            preds = model(img)
            preds = nn.functional.softmax(preds, dim=1)
            loss_value = loss(preds, gt_label)

            loss_value.backward()
            optimizer.step()

            train_loss += loss_value.item()
            _, predicted = preds.max(1)
            total += gt_label.size(0)
            correct += predicted.eq(gt_label).sum().item()

        print("Train acc: %f"%(correct/total))
        print("Train loss: %f"%(train_loss/total))
        writer.add_scalar('train/loss', train_loss/total, epoch)
        writer.add_scalar('train/acc', correct/total, epoch)

        scheduler.step()

        if epoch % 10 == 0:
            val_acc = val(model, testloader, metadata.classes)
            print("Val acc: %f"%(val_acc))
            writer.add_scalar('val/acc', val_acc, epoch)

            if val_acc > best_val_score:
                save_train_model(model, optimizer, epoch, scheduler, weight_file)
            save_train_model(model, optimizer, epoch, scheduler, os.path.join("./runs/%s"%exp_folder, "last_save.pth"))

    writer.close()

# (for the normal CNN)
def freeze_pretrained(model):
    for name, params in model.named_parameters():
        if not 'fc' in name:
            params.requires_grad = False

if __name__ == '__main__':
    
    opt = parseArgs()

    # no pretrained weights:
    if opt.model == 'darknet53':
        # create model here:
        model = Darknet53(10)
        model.apply(weights_init)
    elif opt.model == 'msa_darknet53':
        # create MSA version of darknet:
        model = MSADarknet53(10)
    train(model, opt)