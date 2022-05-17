import matplotlib.pyplot as plt
import pandas
import os 

FOLDER = "dog_map_0_5"
TITLES = ["YOLOv3", "YOLOv3 MSA 4", "YOLOv3 MSA 3", "YOLOv3 MSA 2", "YOLOv3 MSA 1"]
CSV_FILES = [
    "run-train_exp3_dog_vanilla-tag-metrics_mAP_0.5.csv",
    "run-train_dog_yolov3_msa_redux-tag-metrics_mAP_0.5.csv",
    "run-train_dog_yolov3_msa_redux_2-tag-metrics_mAP_0.5.csv",
    "run-train_dog_yolov3_msa_redux_3-tag-metrics_mAP_0.5.csv",
    "run-train_dog_yolov3_msa_redux_4-tag-metrics_mAP_0.5.csv",
]

#fig = plt.figure()
fig, ax_list = plt.subplots(1,2, figsize=(10,5))
for _file, _title in zip(CSV_FILES, TITLES):
    df = pandas.read_csv(os.path.join(FOLDER, _file))
    _map = df['Value'].to_numpy()
    _x = [i for i in range(len(_map))]
    print(_map)
    ax_list[0].plot(_x, _map, label=_title)
    ax_list[1].plot(_x, _map, label=_title)
ax_list[1].set_ylim([0.875,0.93])
ax_list[1].set_xlim([50,355])
plt.legend()
ax_list[0].set_title('All YOLOv3 Configurations\nOxford IIT Pet mAP')
ax_list[1].set_title('All YOLOv3 Configurations\nOxford IIT Pet mAP (Zoomed)')
ax_list[0].set_ylabel('mAP @ 0.5')
ax_list[0].set_xlabel('Epochs')
ax_list[1].set_ylabel('mAP @ 0.5')
ax_list[1].set_xlabel('Epochs')
#plt.yscale('log')
plt.tight_layout()
plt.savefig('./graphs/all_dog_maps.png')


#####################

FOLDER = "voc_map_0_5"
TITLES = ["YOLOv3", "YOLOv3 MSA 4", "YOLOv3 MSA 3", "YOLOv3 MSA 2", "YOLOv3 MSA 1"]
CSV_FILES = [
    "run-train_voc2007_yolov3_vanilla-tag-metrics_mAP_0.5.csv",
    "run-train_voc2007_yolov3_redux-tag-metrics_mAP_0.5.csv",
    "run-train_voc2007_yolov3_msa_redux_2-tag-metrics_mAP_0.5.csv",
    "run-train_voc2007_yolov3_msa_redux_3-tag-metrics_mAP_0.5.csv",
    "run-train_voc2007_yolov3_msa_redux_4-tag-metrics_mAP_0.5.csv",
]

#fig = plt.figure()
fig, ax_list = plt.subplots(1,2, figsize=(10,5))
for _file, _title in zip(CSV_FILES, TITLES):
    df = pandas.read_csv(os.path.join(FOLDER, _file))
    _map = df['Value'].to_numpy()
    _x = [i for i in range(len(_map))]
    print(_map)
    ax_list[0].plot(_x, _map, label=_title)
    ax_list[1].plot(_x, _map, label=_title)
ax_list[1].set_ylim([0.7, 0.78])
ax_list[1].set_xlim([18,60])
plt.legend()
ax_list[0].set_title('All YOLOv3 Configurations\nVOC2007 mAP')
ax_list[1].set_title('All YOLOv3 Configurations\nVOC2007 mAP (Zoomed)')
ax_list[0].set_ylabel('mAP @ 0.5')
ax_list[0].set_xlabel('Epochs')
ax_list[1].set_ylabel('mAP @ 0.5')
ax_list[1].set_xlabel('Epochs')
plt.tight_layout()
#plt.yscale('log')
plt.savefig('./graphs/all_voc_maps.png')