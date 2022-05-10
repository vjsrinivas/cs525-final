import numpy as np
import torch

def saltpepper(image, prob=0.01):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    image = image.cpu().numpy()
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='float32')
            white = np.array([255, 255, 255], dtype='float32')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='float32')
            white = np.array([255, 255, 255, 255], dtype='float32')

    probs = np.random.random(image.shape[:2])
    image[probs < (prob / 2)] = black
    image[probs > 1 - (prob / 2)] = white
    
    image = torch.Tensor(image)
    if torch.cuda.is_available():
        image = image.cuda()
    
    return image