#!/bin/sh
cd models/detectors/yolov3

# 350 epochs for vanilla yolov3
python3 train.py --img 416 --batch 16 --epochs 350 --data oxford_pet.yaml --weights '' --cfg models/yolov3-msa.yaml --adam

# YOLOv3 MSAs (4,3,2,1):
python3 train.py --img 416 --batch 16 --epochs 350 --data oxford_pet.yaml --weights '' --cfg models/yolov3-msa-redux.yaml --adam --hyp data/hyps/hyp.scratch-msa.yaml --name dog_yolov3_msa_redux
python3 train.py --img 416 --batch 16 --epochs 350 --data oxford_pet.yaml --weights '' --cfg models/yolov3-msa-redux_2.yaml --adam --hyp data/hyps/hyp.scratch-msa.yaml --name dog_yolov3_msa_redux_2
python3 train.py --img 416 --batch 16 --epochs 350 --data oxford_pet.yaml --weights '' --cfg models/yolov3-msa-redux_3.yaml --adam --hyp data/hyps/hyp.scratch-msa.yaml --name dog_yolov3_msa_redux_3
python3 train.py --img 416 --batch 16 --epochs 350 --data oxford_pet.yaml --weights '' --cfg models/yolov3-msa-redux_4.yaml --adam --hyp data/hyps/hyp.scratch-msa.yaml --name dog_yolov3_msa_redux_4

# NOT INCLUDED IN FINAL REPORT - APPLY MSA HEAD TO YOLO HEAD (doesn't help):
python3 train.py --img 416 --batch 16 --epochs 350 --data oxford_pet.yaml --weights '' --cfg models/yolov3-msa-head_1.yaml --adam --hyp data/hyps/hyp.scratch-msa.yaml --name dog_yolov3_msa_head_1

# validation data on the trained models:
python val.py --data oxford_pet.yaml --weights runs/train/exp3_dog_vanilla/weights/best.pt --img 416 --name dog_vanilla
python val.py --data oxford_pet.yaml --weights runs/train/dog_yolov3_msa_redux/weights/best.pt --img 416 --name dog_yolov3_msa_redux
python val.py --data oxford_pet.yaml --weights runs/train/dog_yolov3_msa_redux_2/weights/best.pt --img 416 --name dog_yolov3_msa_redux_2
python val.py --data oxford_pet.yaml --weights runs/train/dog_yolov3_msa_redux_3/weights/best.pt --img 416 --name dog_yolov3_msa_redux_3
python val.py --data oxford_pet.yaml --weights runs/train/dog_yolov3_msa_redux_4/weights/last.pt --img 416 --name dog_yolov3_msa_redux_4