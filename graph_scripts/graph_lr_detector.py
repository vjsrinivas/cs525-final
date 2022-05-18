# THIS GRAPHING CODE TAKES IN CSV FILES FOR ALL THE mAP FUNCTIONS FOR EACH NETWORK:

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CSV_FILE = 'lr_obj_detectors.csv'
NPY_FILE = 'lr_dk53.npy'
df = pd.read_csv(CSV_FILE)
_values = df['Value'].to_numpy()
print(_values)

_values_cifar = np.load(NPY_FILE, allow_pickle=True)
_x = [i for i in range(len(_values))]
_x_cifar = [i for i in range(len(_values_cifar))]
fig, ax_list = plt.subplots(1,2, figsize=(10,5))
ax_list[0].plot(_x, _values)
ax_list[0].set_ylabel("Learning Rate Value")
ax_list[0].set_xlabel("Epochs")
ax_list[0].set_title('Learning Rate Scheduler\nfor YOLOv3')
ax_list[1].plot(_x_cifar, _values_cifar)
ax_list[1].set_ylabel("Learning Rate Value")
ax_list[1].set_xlabel("Epochs")
ax_list[1].set_title('Learning Rate Scheduler\nfor Darknet53')
plt.tight_layout()
plt.savefig('./graphs/lr_models.png')