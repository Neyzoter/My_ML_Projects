# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 20:27:11 2018

Housing Prices Competition for Kaggle Learn Users

@author: HP528

@version:0.0.1
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing.imputation import Imputer


train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")

# 获取训练集
train_target = train_data['SalePrice']
train_features = train_data.drop(['Id','SalePrice'],axis = 1) 

# 获取测试集
test_features = test_data.drop(['Id'],axis = 1)

## 获取有缺失值的列
#missing_cols_train = [col for col in train_features.columns
#                     if train_features[col].isnull().any()]
#
#missing_cols_test = [col for col in test_features.columns
#                     if test_features[col].isnull().any()]
#
###查找test和train中包含NaN的列有什么不同
##[col for col in missing_cols_train if col not in missing_cols_test]
#'''
#test NaN 而train没有的
#['MSZoning',
# 'Utilities',
# 'Exterior1st',
# 'Exterior2nd',
# 'BsmtFinSF1',
# 'BsmtFinSF2',
# 'BsmtUnfSF',
# 'TotalBsmtSF',
# 'BsmtFullBath',
# 'BsmtHalfBath',
# 'KitchenQual',
# 'Functional',
# 'GarageCars',
# 'GarageArea',
# 'SaleType']
#
#train NaN而test没有的
#['Electrical']
#'''
#
#missing_cols = missing_cols_train+missing_cols_test # 在训练集和测试集中有缺失值的列
### 45个特征
#train_features.drop(missing_cols,axis = 1,inplace = True)
#test_features.drop(missing_cols,axis = 1,inplace = True)
missing_cols_train = [col for col in train_features.columns
                     if train_features[col].isnull().any()]
print('missing features:'+str(missing_cols_train))
# 独热
train_features = pd.get_dummies(train_features)
test_features = pd.get_dummies(test_features)

missing_cols_train = [col for col in train_features.columns
                     if train_features[col].isnull().any()]
print('missing features:'+str(missing_cols_train))

#print("train_features's num: ",train_features.columns.size)
#print("test_features's num: ",test_features.columns.size)
#print("feature's name: ",[col for col in test_features.columns
#                          if col not in train_features.columns])

train_features,test_features = train_features.align(test_features,
                                                    join='left',
                                                    axis = 1)
missing_cols_train = [col for col in train_features.columns
                     if train_features[col].isnull().any()]
print('missing features:'+str(missing_cols_train))
#print(train_features.LotFrontage)
# 缺失值处理
my_imputer = Imputer(strategy = 'median')
train_features = my_imputer.fit_transform(train_features)
test_features = my_imputer.transform(test_features)
#print(train_features.LotFrontage)
##print("features num : "+len(train_features.columns))
## 训练数据集分割成训练集和测试集，用于测试

X_train, X_test, y_train, y_test = train_test_split(train_features, 
                                                    train_target,
                                                    train_size=0.8, 
                                                    test_size=0.2, 
                                                    random_state=0)

# 训练XGBOOST
model = XGBRegressor()
model.fit(X_train,y_train, verbose=False)

predictions = model.predict(X_test)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))

# 测试集输出
test_predictions = model.predict(test_features)


my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_predictions})
my_submission.to_csv('submission.csv', index=False)


