import os
import sys
from tqdm.notebook import trange, tqdm

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets

import torch.optim as optim
from models.classifiers.darknet import msa 

#Conv block:
def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())

# Residual Block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out

# Build Darknet53 with MSA (attention block basic A)
class Darknet53(nn.Module):
    def __init__(self, num_classes):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes
        self.block = DarkResidualBlock

        self.conv1 = conv_batch(3, 32)
        
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(self.block, in_channels=64, num_blocks=1)
        
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(self.block, in_channels=128, num_blocks=1)
        self.attention_block2 = msa.AttentionBasicBlockA(128, window_size=8, heads=3)

        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(self.block, in_channels=256, num_blocks=7)
        self.attention_block3 = msa.AttentionBasicBlockA(256, window_size=4, heads=6)
        
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(self.block, in_channels=512, num_blocks=7)
        self.attention_block4 = msa.AttentionBasicBlockA(512, window_size=2, heads=12)
        
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(self.block, in_channels=1024, num_blocks=3)
        self.attention_block5 = msa.AttentionBasicBlockA(1024, window_size=1, heads=24)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)

    # feedforward behavior
    def forward(self, x):
        out = self.conv1(x)
        
        out = self.conv2(out)
        out = self.residual_block1(out)
        
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.attention_block2(out)

        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.attention_block3(out)

        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.attention_block4(out)

        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.attention_block5(out)
        
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)
        return out

    # make each stage
    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

    def load_custom_state_dict(self, weights):
        try:
            self.load_state_dict(weights, strict=False) # CAREFUL!!! DANGEROUS IF NOT HANDLED CORRECTLY
        except Exception as e:
            print(e)
