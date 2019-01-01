# -*- coding: utf-8 -*-
"""
Housing Prices Competition for Kaggle Learn Users
@author: 宋超超
@version:0.0.1
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing.imputation import Imputer
# 读取csv文件数据
train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")
# 获取训练集
train_target = train_data['SalePrice']
train_features = train_data.drop(['Id','SalePrice'],axis = 1) 
# 获取测试集
test_features = test_data.drop(['Id'],axis = 1)
# 获取缺失特征
missing_cols_train = [col for col in train_features.columns
                     if train_features[col].isnull().any()]
print('Original missing features:'+str(missing_cols_train))
# 独热码
train_features = pd.get_dummies(train_features)
test_features = pd.get_dummies(test_features)
# 联结所有特征
train_features,test_features = train_features.align(test_features,
                                                    join='outer',
                                                    axis = 1)
# 联结所有特征后造成缺失值
missing_cols_train = [col for col in train_features.columns
                     if train_features[col].isnull().any()]
print('Aligned missing features:'+str(missing_cols_train))
# 缺失值处理
my_imputer = Imputer(strategy = 'median')
train_features = my_imputer.fit_transform(train_features)
test_features = my_imputer.transform(test_features)
# 训练数据集分割成训练集和测试集，用于测试
X_train, X_test, y_train, y_test = train_test_split(train_features, 
                                                    train_target,
                                                    train_size=0.8, 
                                                    test_size=0.2, 
                                                    random_state=0)
# 训练XGBOOST
model = XGBRegressor(max_depth=7,learning_rate=0.1,Missing=None)
model.fit(X_train,y_train, verbose=False)
# 预测测试集
predictions = model.predict(X_test)
# 平均绝对误差
print("Mean absolute error : " + str(mean_absolute_error(predictions, y_test)))
# 目标集输出
test_predictions = model.predict(test_features)
my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_predictions})
my_submission.to_csv('submission.csv', index=False)


