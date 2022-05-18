cd models/detectors/yolov3

# 57 epochs for vanilla yolov3
python3 train.py --img 416 --batch 16 --epochs 57 --data voc.yaml --weights '' --cfg models/yolov3.yaml --adam --hyp data/hyps/hyp.scratch-msa.yaml --workers 4 --name voc2007_yolov3_vanilla

# YOLOv3 MSAs (4,3,2,1):
python3 train.py --img 416 --batch 16 --epochs 57 --data voc.yaml --weights '' --cfg models/yolov3-msa-redux.yaml --adam --hyp data/hyps/hyp.scratch-msa.yaml --workers 4 --name voc2007_yolov3_redux
python3 train.py --img 416 --batch 16 --epochs 57 --data voc.yaml --weights '' --cfg models/yolov3-msa-redux_2.yaml --adam --hyp data/hyps/hyp.scratch-msa.yaml --workers 4 --name voc2007_yolov3_msa_redux_2
python3 train.py --img 416 --batch 16 --epochs 57 --data voc.yaml --weights '' --cfg models/yolov3-msa-redux_3.yaml --adam --hyp data/hyps/hyp.scratch-msa.yaml --workers 4 --name voc2007_yolov3_msa_redux_3
python3 train.py --img 416 --batch 16 --epochs 57 --data voc.yaml --weights '' --cfg models/yolov3-msa-redux_4.yaml --adam --hyp data/hyps/hyp.scratch-msa.yaml --workers 4 --name voc2007_yolov3_msa_redux_4

# validation data on the trained models:
python val.py --data voc.yaml --weights runs/train/voc2007_yolov3_vanilla/weights/best.pt --img 416 --name voc2007_yolov3_vanilla
python val.py --data voc.yaml --weights runs/train/voc2007_yolov3_redux/weights/best.pt --img 416 --name voc2007_yolov3_redux
python val.py --data voc.yaml --weights runs/train/voc2007_yolov3_msa_redux_2/weights/best.pt --img 416 --name voc2007_yolov3_msa_redux_2
python val.py --data voc.yaml --weights runs/train/voc2007_yolov3_msa_redux_3/weights/best.pt --img 416 --name voc2007_yolov3_msa_redux_3
python val.py --data voc.yaml --weights runs/train/voc2007_yolov3_msa_redux_4/weights/best.pt --img 416 --name voc2007_yolov3_msa_redux_4