cd models/detectors/yolov3
IMG=000111
python3 viz.py \
    --weights runs/train/voc2007_yolov3_redux/weights/best.pt \
    --source ../datasets/VOC/images/test2007/$IMG.jpg \
    --img 416 \
    --device cpu \
    --prefix msa1 \
    --attn_loc 14

python3 viz.py \
    --weights runs/train/voc2007_yolov3_msa_redux_2/weights/best.pt \
    --source ../datasets/VOC/images/test2007/$IMG.jpg \
    --img 416 \
    --device cpu \
    --prefix msa2 \
    --attn_loc 13

python3 viz.py \
    --weights runs/train/voc2007_yolov3_msa_redux_3/weights/best.pt \
    --source ../datasets/VOC/images/test2007/$IMG.jpg \
    --img 416 \
    --device cpu \
    --prefix msa3 \
    --attn_loc 12

python3 viz.py \
    --weights runs/train/voc2007_yolov3_msa_redux_4/weights/best.pt \
    --source ../datasets/VOC/images/test2007/$IMG.jpg \
    --img 416 \
    --device cpu \
    --prefix msa4 \
    --attn_loc 11

python3 viz_graph.py