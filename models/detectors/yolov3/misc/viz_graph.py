import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ORIGINAL_PATH = './runs/viz/'
    IMAGES = ['000013', '000111', '009889', '007761']
    MODEL_NAMES = ['msa1', 'msa2', 'msa3', 'msa4']
    MODEL_TITLES = ['YOLOv3 MSA 4', 'YOLOv3 MSA 3', 'YOLOv3 MSA 2', 'YOLOv3 MSA 1'] # must be corresponding to MODEL_NAMES
    TYPES = ['image','upsampled']

    for _img in IMAGES:
        fig, ax_list = plt.subplots(len(TYPES),len(MODEL_NAMES), figsize=(7,3))
        for i,(model_name, model_title) in enumerate(zip(MODEL_NAMES, MODEL_TITLES)):
            for j,_type in enumerate(TYPES):
                _npy_file = "%s_%s_%s.npy"%(_img, model_name, _type)
                _npy_file = os.path.join(ORIGINAL_PATH, _npy_file)
                assert os.path.exists(_npy_file), "%s not found"%(_npy_file)
                _data = np.load(_npy_file, allow_pickle=True)
                if _type == 'image':
                    _data = cv2.cvtColor(_data, cv2.COLOR_RGB2BGR)    
                    ax_list[j][i].set_title(model_title)
                print(_data.shape, _type)   

                ax_list[j][i].imshow(_data)
                ax_list[j][i].set_yticks([])
                ax_list[j][i].set_xticks([])
        plt.tight_layout()
        plt.show()
