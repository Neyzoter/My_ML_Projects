import keras.backend as K
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
import scipy
from scipy import ndimage

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    
def load_myFaces(img_file,num_px,num_py):
    img1 = np.array(ndimage.imread(img_file, flatten=False))   
    img2 = scipy.misc.imresize(img1, size=(num_px,num_py))
    return img2

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    X_input = Input(shape=input_shape) # 输入数据的形式,一幅图
    X = ZeroPadding2D(padding=(1, 1))(X_input) # 0 pad
    X = Conv2D(8, kernel_size=(3,3), strides=(1,1))(X)  # Conv2D(过滤器个数, kernel_size, strides)(X)
    X = BatchNormalization(axis=3)(X)  # 归一化，axis：坐标个数
    X = Activation('relu')(X)  # 激活函数
    X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(X)  # 最大池化层，valid：不填充pad
    
    X = ZeroPadding2D(padding=(1, 1))(X)  # 0 pad
    X = Conv2D(16, kernel_size=(3,3), strides=(1,1))(X) # 16个过滤器
    X = BatchNormalization(axis=3)(X) # 归一化
    X = Activation('relu')(X) # 激活函数
    X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(X)  # 最大池化层，valid：不填充pad
    
    X = ZeroPadding2D(padding=(1, 1))(X)  # 0pad
    X = Conv2D(32, kernel_size=(3,3), strides=(1,1))(X) # 32个过滤器
    X = BatchNormalization(axis=3)(X) # 归一化
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(X)
    
    # FC
    X = Flatten()(X)
    # Dense(units,activation,...)，其中units表示输出维度
    Y = Dense(1, activation='sigmoid')(X)
    
    model = Model(inputs = X_input, outputs = Y, name='HappyModel')
    
    return model