
# 如果没有Nelearn这个文件夹和你的py主函数文件放在一起，则需要把Nelearn的绝对路径加进去
import sys
sys.path.append(r'.') # 这里包含了当前文件夹


import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from Nelearn.extraction.one_hot import getOneHotMatrix_np
from Nelearn.module.simple_NN import getParafromMinibatchModel
from Nelearn.module.simple_NN import simple_NN_predict
from Nelearn.extraction.load_datas import load_dataset_h5


# 下载数据(图片张数,横坐标像素个数,纵坐标像素个数,颜色通道数3)
# 这里把数据集放在了和本py文件一个目录下，所以load_dataset_h5只要输入一个空白的字符串''即可寻找到数据集
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset_h5("./")
# 把数据集扁平化
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

# 归一化数据
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.

# Y one hot一下
Y_train = getOneHotMatrix_np(Y_train_orig, 6)
Y_test = getOneHotMatrix_np(Y_test_orig, 6)

# 训练参数
# num_epochs训练的次数（把所有的数据都训练一遍，而不是一次minibatch），minibatch_size：minibatch的大小
# print_cost：是否打印损失函数的值
parameters = getParafromMinibatchModel(X_train, Y_train, X_test,Y_test,learning_rate = 0.0001,
				num_epochs = 1500, minibatch_size = 32, print_cost = True)

# 你的图片名称
my_image_name = "gesture_3.jpg"

# 把你的图片放到目录image中
fname = "images/" + my_image_name
# 如果flatten=True，则让颜色转化为单一灰度层
image = np.array(ndimage.imread(fname, flatten=False))
# 用scipy.misc.imresize转化图片
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
my_image_prediction = simple_NN_predict(my_image, parameters)

# 打印图片
plt.imshow(image)
# 预测数字
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))