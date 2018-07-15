# 说明
该项目基于我的另外一个项目[https://github.com/Neyzoter/Nelearn](https://github.com/Neyzoter/Nelearn "Nelearn")（为了更好的使用TensorFlow而做的一个python3.6包），实现两个隐藏层的神经网络。这里采用的是2018-4-30版本的Nelearn，已经包含在这个项目中，不需要另外下载。如果想要查看最新版本的可以进入链接查看。

# 系统说明

作用：双隐藏层神经网络识别手势。

模型：双隐藏层神经网络

策略：对数损失函数

算法：Adam

备注：你也可以改成datasets中的其他数据集，去识别其他的物体。

具体可修改/Nelearn/extraction/load_dataset_h5代码。

以下为其他的数据集。

<img src="/images/datasets.png" width="600" alt="其他的数据集" />

Train Accuracy: 1.0

Test Accuracy: 0.791667

# 感谢
该工程和数据集来源于吴恩达老师的深度学习课程。