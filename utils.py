import numpy as np
from PIL import Image

def read_tif(path):
    '''
    :param path: string -> path to the image
    :return: numpy array (range from 0~65536)
    '''

    # open image
    image = Image.open(path)
    # convert it into numpy array
    imarray = np.array(image)
    return  imarray