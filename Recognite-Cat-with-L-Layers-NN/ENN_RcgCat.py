# -*- coding: utf-8 -*-

# 基于L层神经网络（无正则化）的猫咪识别系统

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# 读取训练集和测试集
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
# 读取图片的横纵坐标范围
# 如64表示，图片是64*64像素
num_px = train_x_orig.shape[1]
# 向量化图片数据
(train_x,test_x) = Pics_Reshape(train_x_orig,test_x_orig)

# 神经网络的层数和每层的结点
layers_dims = [12288, 20, 7, 5, 1] #  5-layer model

# 神经网络模型
# 返回参数
parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=True)

# 打印正确率，并返回预测结果
# 1：是猫；0：不是猫
print("\n训练集")
predict(train_x, train_y, parameters)

print("\n测试集")
predict(train_x, train_y, parameters)


print("\n你的图片")
# 自己的图片
my_image = "pig.jpg" 
# 图片是不是猫 1猫；0非猫
my_label_y = [0] 


fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
# 图片用scipy弄成64*64的图片，这样可以使特征数相同
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
# 预测结果，返回1或者0
# 同时输出正确率
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

