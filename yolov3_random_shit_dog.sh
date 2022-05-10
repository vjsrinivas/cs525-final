cd models/detectors/yolov3
python3 train.py --img 416 --batch 16 --epochs 350 --data oxford_pet.yaml --weights '' --cfg models/yolov3-msa.yaml --adam --hyp data/hyps/hyp.scratch-msa.yaml