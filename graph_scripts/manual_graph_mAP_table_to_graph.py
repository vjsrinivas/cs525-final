# THIS GRAPHING CODE TAKES IN CSV FILES FOR ALL THE mAP FUNCTIONS FOR EACH NETWORK:

import numpy as np
# quick and dirty graph from table of mAPs for Dog and VOC:
# MUST BE IN ORDER TO THE TITLES:
TITLES = ["YOLOv3 MSA 4", "YOLOv3 MSA 3", "YOLOv3 MSA 2", "YOLOv3 MSA 1"]
#DOG_MAPS = [0.92, 0.909, 0.928, 0.905, 0.919]
#VOC_MAPS = [0.747, 0.739, 0.726, 0.732, 0.739]
DOG_MAPS = [0.909, 0.928, 0.905, 0.919]
VOC_MAPS = [0.739, 0.726, 0.732, 0.739]
TITLES.reverse()
DOG_MAPS.reverse()
VOC_MAPS.reverse()
DOG_MAPS = np.array(DOG_MAPS)
VOC_MAPS = np.array(VOC_MAPS)
DOG_MAPS -= 0.92
VOC_MAPS -= 0.747

import matplotlib.pyplot as plt
# dog:
num_of_msas = [i+1 for i in range(len(DOG_MAPS))]
fig = plt.figure(figsize=(4,3))
plt.title("mAP Delta on Oxford IIT Pet Testset")
plt.plot(num_of_msas, DOG_MAPS, '-o')
plt.ylabel('mAP Delta from YOLOv3')
plt.xlabel('Number of MSAs')
plt.xticks(num_of_msas)
plt.tight_layout()
#plt.show()
plt.savefig('./graphs/map_delta_dog.svg')


num_of_msas = [i+1 for i in range(len(VOC_MAPS))]
fig = plt.figure(figsize=(4,3))
plt.title("mAP Delta on VOC2007 Testset")
plt.plot(num_of_msas, VOC_MAPS, '-o')
plt.ylabel('mAP Delta from YOLOv3')
plt.xlabel('Number of MSAs')
plt.xticks(num_of_msas)
plt.tight_layout()
#plt.show()
plt.savefig('./graphs/map_delta_voc.svg')