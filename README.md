# YOLOv3 MSA Networks

**Description:** This is the main code for CS525 Final Project that attempts to apply Multi-Headed Self-Attention (MSA) to Object Detectors. The MSA code is taken from ["How Vision Transformers Work"](https://github.com/xxxnell/how-do-vits-work). Many thanks to the authors for their code and great research work.

## Requirements:
- Python 3.6+
- PyTorch 1.8+
- All packages required by YOLOv3 Ultralytics (refer to models/detectors/yolov3/requirements.txt)

## Training Scripts:
We train against the Oxford IIT Pet Dataset and VOC2007+2012. We are also using YOLOv3 by Ultralytics, which introduces new training functionality such as EMA and an Adam optimizer.

We've made simple scripts to run the following trainings:

- **yolov3_dog_train.sh** - Running all different types of YOLOv3 MSA and original YOLOv3 on **Oxford IIT Pet Dataset**
- **yolov3_voc2007_train.sh** - Running all different types of YOLOv3 MSA and original YOLOv3 on **VOC2007+2012**
- **yolov3_voc2007_NO_AUG.sh** - Runs YOLOv3 MSA 4 and YOLOv3 original with NO augmentations
- **yolov3_voc2007_NO_MOSAIC.sh** - Runs YOLOv3 MSA 4 and YOLOv3 original with NO mosaic 
- **yolov3_voc2007_NO_EMA.sh** - Runs YOLOv3 MSA 4 and YOLOv3 original with NO EMA
- **yolov3_voc2007_NO_MOSAIC_NO_EMA.sh** - Runs YOLOv3 MSA 4 and YOLOv3 original with NO mosaic and NO EMA
- **yolov3_voc2007_viz_all_msas_last_channel.sh** - Not fully done script but tries to visualize the attention map on the last MSA module 

For Darknet53-based training, simply just run the following command:
``python3 train_cifar_10.py``

## Folder Structure:

```
- data
|
- graph_scripts
|
- models
|__
    - classifiers
        |__
            - darknet53 (all the darknet53 related training and validation)
    - detectors
        |__
            - yolov3 (ultralytics)
                |_ runs (stores all the runs for training and validation)
|
- runs (for Darknet53 only)
```


## Experiment Weights

You can download the weights of both Darknet53 and YOLOv3 MSAs here:
- Darknet53 weights: [https://drive.google.com/file/d/18I9rKvCbNlT07gq_MnHnEP-B25DDNgUr/view?usp=sharing](https://drive.google.com/file/d/18I9rKvCbNlT07gq_MnHnEP-B25DDNgUr/view?usp=sharing)
- YOLOv3 weights: [https://drive.google.com/file/d/1zu9f8MgcuDUEygQmeScDHtN9O9Mwffqo/view?usp=sharing](https://drive.google.com/file/d/18I9rKvCbNlT07gq_MnHnEP-B25DDNgUr/view?usp=sharing)