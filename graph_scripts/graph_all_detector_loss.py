# THIS GRAPHING CODE TAKES IN CSV FILES FOR ALL THE LOSS FUNCTIONS FOR EACH NETWORK:
# YOLOv3 and YOLOv3 MSA 1,2,3,4

import matplotlib.pyplot as plt
import pandas
import os 

FOLDER = "dog_all_loss"
TITLES = ["YOLOv3", "YOLOv3 MSA 4", "YOLOv3 MSA 3", "YOLOv3 MSA 2", "YOLOv3 MSA 1"]
CSV_FILES_LOC = [
    "run-train_exp3_dog_vanilla-tag-train_box_loss.csv",
    "run-train_dog_yolov3_msa_redux-tag-train_box_loss.csv",
    "run-train_dog_yolov3_msa_redux_2-tag-train_box_loss.csv",
    "run-train_dog_yolov3_msa_redux_3-tag-train_box_loss.csv",
    "run-train_dog_yolov3_msa_redux_4-tag-train_box_loss.csv",
]
CSV_FILES_CLS = [
    "run-train_exp3_dog_vanilla-tag-train_cls_loss.csv",
    "run-train_dog_yolov3_msa_redux-tag-train_cls_loss.csv",
    "run-train_dog_yolov3_msa_redux_2-tag-train_cls_loss.csv",
    "run-train_dog_yolov3_msa_redux_3-tag-train_cls_loss.csv",
    "run-train_dog_yolov3_msa_redux_4-tag-train_cls_loss.csv",
]
CSV_FILES_OBJECTNESS = [
    "run-train_exp3_dog_vanilla-tag-train_obj_loss.csv",
    "run-train_dog_yolov3_msa_redux-tag-train_obj_loss.csv",
    "run-train_dog_yolov3_msa_redux_2-tag-train_obj_loss.csv",
    "run-train_dog_yolov3_msa_redux_3-tag-train_obj_loss.csv",
    "run-train_dog_yolov3_msa_redux_4-tag-train_obj_loss.csv",
]

fig, ax_list = plt.subplots(1,3,figsize=(12, 4))
for j, (file_loc, file_cls, file_obj, _title) in enumerate(zip(CSV_FILES_LOC, CSV_FILES_CLS, CSV_FILES_OBJECTNESS, TITLES)):
    df_loc_loss = pandas.read_csv(os.path.join(FOLDER, file_loc))
    df_cls_loss = pandas.read_csv(os.path.join(FOLDER, file_cls))
    df_obj_loss = pandas.read_csv(os.path.join(FOLDER, file_obj))
    loc_loss = df_loc_loss['Value'].to_numpy()
    cls_loss = df_cls_loss['Value'].to_numpy()
    obj_loss = df_obj_loss['Value'].to_numpy()
    x = [i for i in range(len(obj_loss))]
    ax_list[0].plot(x, loc_loss, label=_title)
    ax_list[1].plot(x, cls_loss, label=_title)
    ax_list[2].plot(x, obj_loss, label=_title)
ax_list[0].set_title('Localization Loss')
ax_list[1].set_title('Classification Loss')
ax_list[2].set_title('Objectness Loss')

for i in range(3):
    ax_list[i].set_ylabel('Loss Value')
for i in range(3):
    ax_list[i].set_xlabel('Epochs')
plt.legend()
plt.suptitle("All Losses for Oxford IIT Pet", fontweight='bold')
plt.tight_layout()
plt.savefig('./graphs/dog_all_loss.svg')
#plt.show()


#####################################################

FOLDER = "voc_all_loss"
TITLES = ["YOLOv3", "YOLOv3 MSA 4", "YOLOv3 MSA 3", "YOLOv3 MSA 2", "YOLOv3 MSA 1"]
CSV_FILES_LOC = [
    "run-train_voc2007_yolov3_vanilla-tag-train_box_loss.csv",
    "run-train_voc2007_yolov3_redux-tag-train_box_loss.csv",
    "run-train_voc2007_yolov3_msa_redux_2-tag-train_box_loss.csv",
    "run-train_voc2007_yolov3_msa_redux_3-tag-train_box_loss.csv",
    "run-train_voc2007_yolov3_msa_redux_4-tag-train_box_loss.csv",
]
CSV_FILES_CLS = [
    "run-train_voc2007_yolov3_vanilla-tag-train_cls_loss.csv",
    "run-train_voc2007_yolov3_redux-tag-train_cls_loss.csv",
    "run-train_voc2007_yolov3_msa_redux_2-tag-train_cls_loss.csv",
    "run-train_voc2007_yolov3_msa_redux_3-tag-train_cls_loss.csv",
    "run-train_voc2007_yolov3_msa_redux_4-tag-train_cls_loss.csv",
]
CSV_FILES_OBJECTNESS = [
    "run-train_voc2007_yolov3_vanilla-tag-train_obj_loss.csv",
    "run-train_voc2007_yolov3_redux-tag-train_obj_loss.csv",
    "run-train_voc2007_yolov3_msa_redux_2-tag-train_obj_loss.csv",
    "run-train_voc2007_yolov3_msa_redux_3-tag-train_obj_loss.csv",
    "run-train_voc2007_yolov3_msa_redux_4-tag-train_obj_loss.csv",
]

fig, ax_list = plt.subplots(1,3,figsize=(12, 4))
for j, (file_loc, file_cls, file_obj, _title) in enumerate(zip(CSV_FILES_LOC, CSV_FILES_CLS, CSV_FILES_OBJECTNESS, TITLES)):
    df_loc_loss = pandas.read_csv(os.path.join(FOLDER, file_loc))
    df_cls_loss = pandas.read_csv(os.path.join(FOLDER, file_cls))
    df_obj_loss = pandas.read_csv(os.path.join(FOLDER, file_obj))
    loc_loss = df_loc_loss['Value'].to_numpy()
    cls_loss = df_cls_loss['Value'].to_numpy()
    obj_loss = df_obj_loss['Value'].to_numpy()
    x = [i for i in range(len(obj_loss))]
    ax_list[0].plot(x, loc_loss, label=_title)
    ax_list[1].plot(x, cls_loss, label=_title)
    ax_list[2].plot(x, obj_loss, label=_title)
ax_list[0].set_title('Localization Loss')
ax_list[1].set_title('Classification Loss')
ax_list[2].set_title('Objectness Loss')

for i in range(3):
    ax_list[i].set_ylabel('Loss Value')
for i in range(3):
    ax_list[i].set_xlabel('Epochs')
plt.legend()
plt.suptitle("All Training Losses for VOC2007+2012", fontweight='bold')
plt.tight_layout()
plt.savefig('./graphs/voc_all_loss.svg')
#plt.show()