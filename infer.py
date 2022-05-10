import torch
from models.classifiers.darknet.darknet53 import Darknet53
from models.classifiers.darknet.darknet53_msa import Darknet53 as MSADarknet53
import numpy as np

if __name__ == '__main__':
    print("Create model")
    model = MSADarknet53(10)
    #model = Darknet53(10)
    pred = model(torch.Tensor(np.ndarray((1,3,100,100))))
    print(pred.shape)