from models.classifiers.darknet.darknet53 import Darknet53
#from models.classifiers.darknet.darknet53_msa_2 import Darknet53 as Darknet53_MSA
import torch
import numpy as np

model = Darknet53(10).cuda()
test  = torch.Tensor(np.ndarray( (1,3,416,416) )).cuda()
#test  = torch.Tensor(np.ndarray( (1,3,512,512) )).cuda()
model(test)
exit()
model = Darknet53_MSA(10, window_size=13).cuda()
model(test)