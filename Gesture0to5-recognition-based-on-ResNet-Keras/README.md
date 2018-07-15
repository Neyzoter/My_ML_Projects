# 说明
本文件包含残差网络ResNet。


# 系统说明

作用：残差神经网络识别手势。

模型：残差神经网络

CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3  -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

其中 ，CONVBLOCK=convolution_block在小路上有1×1卷积，所以可以实现输出和输出维度不同；IDBLOCK=identity_block在小路上是直达，没有1*1卷积，所以输出和输入维度必须相同。

策略：多分类的对数损失函数

算法：Adam

备注：需要较长的训练时间

Train loss =  0.0963 

Train accuracy =  0.9741

Test  loss =  0.609584419926

Test accuracy =  0.86666667064

# 备注
由于训练集和测试集较大，已删除，如果需要请联系我。

# 感谢
该工程和数据集来源于吴恩达老师的深度学习课程。