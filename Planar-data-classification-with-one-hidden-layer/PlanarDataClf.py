# Package imports
import numpy as np
import matplotlib.pyplot as plt
from myOHLNN import *

# matplotlib inline

X, Y = load_planar_dataset() # 生成数据

# 画出参数在二维平面的位置
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral) # 画出图形

### START CODE HERE ### (≈ 3 lines of code)
shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]  # training set size
### END CODE HERE ###
print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))


print ('\n\n------开始训练神经网络------\n')
plt.figure(figsize=(16, 32))
# 隐藏层的数量
HiddenLayer_Num = [1,2,3,4,5,6,7,8,9,10]
for idx,n_h in enumerate(HiddenLayer_Num):
    plt.subplot(5, 2, idx+1)
    # 模型
    parameters = nn_model(X, Y, n_h, num_iterations = 5000, print_cost = False)
    # 画出边界线
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)

    # 得到预测结果
    predictions = predict(parameters, X)
    print("隐藏层数：{},正确率：{}%".format(n_h,float(np.sum(predictions==Y)/Y.shape[1]*100)))
    













