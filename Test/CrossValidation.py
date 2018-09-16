# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 17:24:14 2018

Cross Validation

交叉验证，这里使用S折交叉验证

（1）将数据分成S个互不相交的大小相同的数据集

（2）选取其中S-1个子集作为训练集，其余的一个子集作为测试集

共有S种选法。分成5个子集，即训练出5个模型的图见Cross-Validation.png

（3）第（2）步迭代S次，完成S个模型的训练

（4）可选取其中平均测试误差最小的模型

@author: HP528
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score


data = pd.read_csv('./melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price

my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())

# cv：即数据集分割后的子集个数，即上面的S。会训练出cv个模型，并获得每个模型的误差
scores = cross_val_score(my_pipeline, X, y, cv = 5,scoring='neg_mean_absolute_error')
print(scores)

print('Mean Absolute Error %2f' %(-1 * scores.mean()))

