import numpy as np
import keras
from keras import layers

from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import HappyModel,load_myFaces,load_dataset

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# 读取数据
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# 归一化特征
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# 将Y编程列向量
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("训练集的样本数 = " + str(X_train.shape[0]))
print ("训练集的样本数 = " + str(X_test.shape[0]))
print ("训练集特征的维度: " + str(X_train.shape))
print ("训练集标签的维度: " + str(Y_train.shape))
print ("测试集特征的维度: " + str(X_test.shape))
print ("测试集标签的维度: " + str(Y_test.shape))

# 创建一个Model模型
happyModel = HappyModel((64, 64, 3))
# 配置学习过程
# lr：学习率
# binary_crossentropy即交叉熵代价函数，C=-1/n(sum(ylna+(1-y)ln(1-a)))
# beta_1是momentum的参数，Vdw=beta_1*dw+(1-beta_1)*dw，db类似。momentum的作用就是采用指数加权平均，减缓下降速度
# beta_2是RMSprop的参数，Sdw=beta_2*Sdw+(1-beta_2)*dw*dw,db类似。RMSProp的作用是如果梯度绝对值变小，则加快速率
# Vdw和Sdw都要经过偏差修正，即分别除以一个(1-pow(beta_1,t))和(1-pow(beta_2,t)),其中t表示Adam运行次数
# 补充：adam如何更新参数：w = w - alpha*(Vdw/(sqrt(Sdw)+epsilon))，其中Vdw和Sdw均为修正后的值，epsilon用于防止Sdw或者Vdw过小
happyModel.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])
# 模型训练
happyModel.fit(x=X_train, y=Y_train, batch_size=16, epochs=5)
# 评估模型的性能
preds = happyModel.evaluate(x=X_test, y=Y_test)
# 打印预测测试集的损失函数和正确率
print ("\n\nLoss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

# 图片的其他函数
happyModel.summary()# 输出模型的细节，比如某一层的参数个数等
plot_model(happyModel, to_file='HappyModel.png') # 画图模型
SVG(model_to_dot(happyModel).create(prog='dot', format='svg')) 

# 我自己的图片
img = []
for i in [1,0]:
    for j in [0,1,2,3]:
        img_file = "./myFace/"+str(i)+"_"+str(j)+".jpg"
        img.append((load_myFaces(img_file,64,64)))
img = np.array(img,dtype=np.float)
img = preprocess_input(img,mode="caffe")
plt.imshow(img[1])
plt.show()
print(np.shape(img))
preds = happyModel.evaluate(x=img, y=np.array([1,1,1,1,0,0,0,0]).T)   
print ("\n\nLoss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

"""
img_path = 'myFace/0_1.jpg'

img = image.load_img(img_path, target_size=(64, 64))# 读取图片，并将其转化为64*64
imshow(img) # 显示图片
x = image.img_to_array(img) # 将PIL的图片格式，转化为np.array格式
x = np.expand_dims(x, axis=0)  # 拓展X的维度，axis=0表示在最前面,axis=1表示在第二个添加1维
x = preprocess_input(x) # 预处理，归一化

print(happyModel.predict(x))
"""
