cd models/detectors/yolov3

# No mosaic enabled (changed in hyperparameters folder) - do training on YOLOv3 vanilla and then YOLOv3 MSA 4:
python3 train.py --img 416 --batch 16 --epochs 57 --data voc.yaml --weights '' --cfg models/yolov3.yaml --adam --hyp data/hyps/hyp.scratch-msa-no-mosaic.yaml --workers 4 --name voc2007_yolov3_vanilla_NO_MOSAIC
python3 train.py --img 416 --batch 16 --epochs 57 --data voc.yaml --weights '' --cfg models/yolov3-msa-redux.yaml --adam --hyp data/hyps/hyp.scratch-msa-no-mosaic.yaml --workers 4 --name voc2007_yolov3_redux_NO_MOSAIC
python val.py --data voc.yaml --weights runs/train/voc2007_yolov3_vanilla_NO_MOSAIC/weights/best.pt --img 416 --name voc2007_yolov3_vanilla_NO_MOSAIC
python val.py --data voc.yaml --weights runs/train/voc2007_yolov3_redux_NO_MOSAIC/weights/best.pt --img 416 --name voc2007_yolov3_redux_NO_MOSAIC
