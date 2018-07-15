**目录**

* [仓库作用](#仓库作用)
* [文件简介](#文件简介)
	* [基于单隐层神经网络的二维数据分类Planar-data-classification-with-one-hidden-layer](#基于单隐层神经网络的二维数据分类)
	* [基于L层简单神经网络的猫图识别Recognite-Cat-with-L-Layers-NN](#基于L层简单神经网络的猫图识别)
	* [基于Nelearn2层神经网络的手势识别Gesture0to5-recognition-based-on-Nelearn-and-tensorflow](#基于Nelearn2层神经网络的手势识别)
	* [基于Keras卷积神经网络的笑脸识别HappyFace-Recognition-Keras](#基于Keras卷积神经网络的笑脸识别)
	* [基于Keras残差网络的手势识别Gesture0to5-recognition-based-on-ResNet-Keras](#基于Keras残差网络的手势识别)
	* [基于YOLOv2的80种物体识别CarDetector-YOLO](###基于YOLOv2的80种物体识别)


# 仓库作用
本仓库用于存储我的小型机器学习项目

# 文件简介
### 基于单隐层神经网络的二维数据分类
**Planar-data-classification-with-one-hidden-layer**

**作用**：分类算法进行（标签）蓝色和红色点、二维特征数据的二分类

**模型**：单层神经网络

**策略**：对数损失函数

**算法**：梯度下降法

**备注**：无正则化，可能有过拟合

### 基于L层简单神经网络的猫图识别
**Recognite-Cat-with-L-Layers-NN**

**作用**：神经网络分类算法进行图像识别猫咪

**模型**：L层神经网络

**策略**：对数损失函数

**算法**：梯度下降法

**备注**：无正则化，可能有过拟合；系统识别猫的效果不太好，区别不了猫和其他图片

### 基于Nelearn(TensorFlow)2层神经网络的手势识别
**Gesture0to5-recognition-based-on-Nelearn-and-tensorflow**

**作用**：双隐藏层神经网络识别手势。

**模型**：双隐藏层神经网络

**策略**：对数损失函数

**算法**：Adam

**备注**：你也可以改成datasets中的其他数据集，去识别其他的物体。

Train Accuracy: 1.0

Test Accuracy: 0.791667

### 基于Keras卷积神经网络的笑脸识别
**HappyFace-Recognition-Keras**

**作用**：卷积神经网络识别笑脸。

**模型**：卷积神经网络

**策略**：交叉熵损失函数-1/n(ylog(a)-(1-y)log(a))

**算法**：Adam

**备注**：训练集不够大，当对自己的照片进行识别时，效果不佳。

Train loss =  0.0120

Train accuracy =  0.9983

Test  loss =  0.102046815952

Test accuracy =  0.980000003974

### 基于Keras残差网络的手势识别
**Gesture0to5-recognition-based-on-ResNet-Keras**

**作用**：残差神经网络识别手势。

**模型**：残差神经网络

CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3  -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

其中 ，CONVBLOCK=convolution_block在小路上有1×1卷积，所以可以实现输出和输出维度不同；IDBLOCK=identity_block在小路上是直达，没有1*1卷积，所以输出和输入维度必须相同。

**策略**：多分类的对数损失函数

**算法**：Adam

**备注**：需要较长的训练时间

Train loss =  0.0963 

Train accuracy =  0.9741

Test  loss =  0.609584419926

Test accuracy =  0.86666667064

### 基于YOLOv2的80种物体识别
**CarDetector-YOLO**

作用：基于YOLOv2的80种物体识别。

模型：残差神经网络

策略：？

算法：？

备注：直接下载了已经训练好的模型



