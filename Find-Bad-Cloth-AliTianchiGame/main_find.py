# -*- coding: utf-8 -*-
import numpy as np
from resnets_utils import ResNet50,load_dataset,convert_to_one_hot
import scipy.misc
from matplotlib.pyplot import imshow
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

bad_pics_path = ".\\dataset\\train\\bad\\"

ok_pics_path = ".\\dataset\\train\\ok\\"

