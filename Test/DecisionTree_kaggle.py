# -*- coding: utf-8 -*-
"""
决策树
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor  # 决策树
from sklearn.model_selection import train_test_split # 训练集和测试集分离
from sklearn.metrics import mean_absolute_error  # 平均绝对误差

file_path = './melb_data.csv'

melbourne_data = pd.read_csv(file_path)

print(melbourne_data.columns) # 打印每列的名称

melbourne_data = melbourne_data.dropna(axis=0)  # 去掉缺失值

# 得到输出值
y = melbourne_data['Price']


# 获取特征
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(X.describe())  # 输出每列数据的个数、平均值、最大最小等方面

print(X.head())  # 输出前几个

# 分裂出测试集和训练集
train_X,val_X,train_Y,val_Y = train_test_split(X,y,random_state = 0)

'''
1.Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
2.Fit: Capture patterns from provided data. This is the heart of modeling.
3.Predict: Just what it sounds like
4.Evaluate: Determine how accurate the model's predictions are.
'''

# define
# max_leaf_nodes最大叶子节点个数，太大可能会造成过拟合
# 可以通过evaluate评估多个叶子结点个数的模型的误差来确定最好的那个
melbourne_model = DecisionTreeRegressor(max_leaf_nodes=100,random_state=1)
# fit
melbourne_model.fit(train_X,train_Y)
# predict
pred_Y = melbourne_model.predict(val_X)

# evaluate
print(mean_absolute_error(val_Y, pred_Y))
