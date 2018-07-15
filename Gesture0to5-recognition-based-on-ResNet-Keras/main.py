import numpy as np
from resnets_utils import ResNet50,load_dataset,convert_to_one_hot
import scipy.misc
from matplotlib.pyplot import imshow
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

# 数据提取
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


# 建立模型，输入格式是64,64,3)，6个分类
model = ResNet50(input_shape = (64, 64, 3), classes = 6)

# 编译模型设置，adam优化器，多分类交叉熵损失函数，
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练
model.fit(X_train, Y_train, epochs = 20, batch_size = 32)

# 预测
preds = model.evaluate(X_test,Y_test)

print("loss = ",preds[0])
print("accuracy = ",preds[1])

model.summary()
plot_model(model, to_file='model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))