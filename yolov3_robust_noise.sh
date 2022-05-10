#!/bin/sh

# yolov3-msa -> exp2
cd models/detectors/yolov3
python3 degrade.py \
    --imgsz 416 \
    --weights runs/train/exp2_dog_msa/weights/best.pt \
    --data oxford_pet.yaml \
    --noise salt_and_pepper \
    --name yolov3_msa_salt_pepper

# yolov3 (vanilla) -> exp3
python3 degrade.py \
    --imgsz 416 \
    --weights runs/train/exp3_dog_vanilla/weights/best.pt \
    --data oxford_pet.yaml \
    --noise salt_and_pepper \
    --name yolov3_salt_pepper

