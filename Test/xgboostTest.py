# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:38:20 2018

xgboost

@author: SongChaochao
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor

data = pd.read_csv('./train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)


# xgboost模型
my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)

# make predictions
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))


#### XGBOOST的超参数
## n_estimators：运行模型的次数，过多：过拟合；过少：欠拟合
# learning_rate:学习率
# n_job：使用并行计算，一般机器有几个内核就取几
my_model = XGBRegressor(n_estimators=1000,learning_rate=0.05,n_job = 4)
# early_stopping_rounds：提早结束迭代，即连续early_stopping_rounds次出现交叉验证分数恶化，则停止迭代
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)