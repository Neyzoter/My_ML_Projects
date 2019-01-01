import os
import random
import numpy as np
import tensorflow as tf
import math

from PIL import Image
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform

import xml.dom.minidom

def random_read_bad_picsnum(picsnum=32):
    """
    随机读取100张坏布的图片
    """
    xml_path = ".\\datasets\\train\\bad\\xml\\"
    pic_path = ".\\datasets\\train\\bad\\pics\\"
    pathDir = os.listdir(xml_path)
    samples = random.sample(pathDir, picsnum)
    features = []
    for name in samples: 
        # dom = xml.dom.minidom.parse("./datasets/xuelang_round1_train_part1_20180628/吊纬/J01_2018.06.13 13_25_43.xml")
        dom = xml.dom.minidom.parse(xml_path+name)
        root = dom.documentElement
#        print(root.nodeName)
#        print(root.nodeValue)
#        print(root.nodeType)
#        print(root.ELEMENT_NODE)
        
        ######   ----XML读取范例----  ######
        # 获得子标签
        # 如果有多个相同的标签，则都会读出来，用[]索引即可
#        tagname_root_size = root.getElementsByTagName('size')[0]
        # 取第一个标签（这里也只有一个）
#        print(tagname_root_size.nodeName)
        
        # 获取属性，不过这个文件中的<size>没有属性
        # 如果有,比如<size id = '123'>数值<\size>
        # tagname_root_size_width = tagname_root_size[0].getAttribute('id')
        
        # 得到size下面的长和宽
#        tagname_root_size_width = tagname_root_size.getElementsByTagName('width')[0]
#        print(tagname_root_size_width.firstChild.data) #  读取width
#        tagname_root_size_height = tagname_root_size.getElementsByTagName('height')[0]
#        print(tagname_root_size_height.firstChild.data) #  读取width
        
        # 读取该布的问题名称
        tagname_root_object = root.getElementsByTagName('object')[0]
#        tagname_root_object_name = tagname_root_object.getElementsByTagName('name')[0]
#        print(tagname_root_object_name.firstChild.data)
        
        # 读取box的对角坐标
        tagname_root_object_bndbox = tagname_root_object.getElementsByTagName('bndbox')[0]
        tagname_root_object_bndbox_xmin = int(tagname_root_object_bndbox.getElementsByTagName('xmin')[0].firstChild.data)
        tagname_root_object_bndbox_ymin = int(tagname_root_object_bndbox.getElementsByTagName('ymin')[0].firstChild.data)
        tagname_root_object_bndbox_xmax = int(tagname_root_object_bndbox.getElementsByTagName('xmax')[0].firstChild.data)
        tagname_root_object_bndbox_ymax = int(tagname_root_object_bndbox.getElementsByTagName('ymax')[0].firstChild.data)

        lst =[tagname_root_object_bndbox_xmin,tagname_root_object_bndbox_ymin,tagname_root_object_bndbox_xmax,tagname_root_object_bndbox_ymax,0] # 0表示bad
#        print(lst)
        features.append(lst)
        
        img=Image.open(pic_path+name)
        
    return np.array(features)
        
    


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


# 输入输出的维度可以不同
def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # 定义名字
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # 过滤器的个数
    F1, F2, F3 = filters
    
    # 保存输入值
    X_shortcut = X


    ##### 主要路线 #####
    # Conv2D返回一个函数
    # 第一个卷积，F1：卷积核的个数即输出维度，1*1卷积，步长可配置（默认为2），采用valid pad，kernel_initializer权值初始化的方法为Glorot均匀分布初始化方法
    # glorot_uniform：参数从[-limit, limit]的均匀分布产生，limit=sqrt(6 / (fan_in + fan_out))，其中fan_in权重张量的输入单元数，fan_out权重张量的输出单元数
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (s,s), name = conv_name_base + '2a', padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X) # 归一化
    X = Activation('relu')(X) # 激活函数
    

    # 第二个卷积，f*f卷积，same pad不改变输入特征的2d大小，采用Glorot均匀分布初始化方法
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), name = conv_name_base + '2b',padding='same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # 第三个卷积
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), name = conv_name_base + '2c',padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s, s), name = conv_name_base + '1',padding='valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # 主路和小路的特征组合
    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)
        
    return X


# identity block：输入输出的维度相同
def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)
      
    return X


# 50 层的残差网络
def ResNet50(input_shape = (2560, 1920, 3), classes = 2):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    
    IDBLOCK = identity_block
    CONVBLOCK = convolutional_block
    
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1:CONV2D -> BATCHNORM -> RELU -> MAXPOOL
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2：CONVBLOCK -> IDBLOCK*2
    # f*f卷积核
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, f = 3, filters = [64, 64, 256], stage=2, block='b')
    X = identity_block(X, f = 3, filters = [64, 64, 256], stage=2, block='c')


    # Stage 3：CONVBLOCK -> IDBLOCK*3
    # The convolutional block uses three set of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".
    # The 3 identity blocks use three set of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".
    X = convolutional_block(X, f = 3, filters=[128,128,512], stage = 3, block='a', s = 2)
    X = identity_block(X, f = 3, filters=[128,128,512], stage= 3, block='b')
    X = identity_block(X, f = 3, filters=[128,128,512], stage= 3, block='c')
    X = identity_block(X, f = 3, filters=[128,128,512], stage= 3, block='d')

    # Stage 4：CONVBLOCK -> IDBLOCK*5
    # The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
    # The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
    X = convolutional_block(X, f = 3, filters=[256, 256, 1024], block='a', stage=4, s = 2)
    X = identity_block(X, f = 3, filters=[256, 256, 1024], block='b', stage=4)
    X = identity_block(X, f = 3, filters=[256, 256, 1024], block='c', stage=4)
    X = identity_block(X, f = 3, filters=[256, 256, 1024], block='d', stage=4)
    X = identity_block(X, f = 3, filters=[256, 256, 1024], block='e', stage=4)
    X = identity_block(X, f = 3, filters=[256, 256, 1024], block='f', stage=4)

    # Stage 5:CONVBLOCK -> IDBLOCK*2
    # The convolutional block uses three set of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".
    # The 2 identity blocks use three set of filters of size [256, 256, 2048], "f" is 3 and the blocks are "b" and "c".
    X = convolutional_block(X, f = 3, filters=[512, 512, 2048], stage=5, block='a', s = 2)
    
    # filters should be [256, 256, 2048], but it fail to be graded. Use [512, 512, 2048] to pass the grading
    X = identity_block(X, f = 3, filters=[256, 256, 2048], stage=5, block='b')
    X = identity_block(X, f = 3, filters=[256, 256, 2048], stage=5, block='c')

    # 平均池化层 AVGPOOL.
    # 2*2 的窗口
    X = AveragePooling2D(pool_size=(2,2))(X)

    # 扁平化，eg.(None, 64, 32, 32) --> (None, 65536)
    X = Flatten()(X)
    # 全连接层
    # 这里的softmax有一点问题
    # keras版本2.1.6，函数softmax()有一个参数为axis，传给tf.nn.softmax()。在tf中是dim，而在keras中用了axis=axis，改成dim=axis即可。
    # 具体的方法   https://neyzoter.github.io/2018/07/13/keras-softmax-err/
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


if __name__ == "__main__":
    features = random_read_bad_picsnum()
    print(type(features[1][1]))
    