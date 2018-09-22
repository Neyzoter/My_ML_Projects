# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:45:37 2018

处理缺失数据

@author: HP528
"""

#(1)Drop Columns with Missing Values
# 适用于很多数值都缺失的

##单个数据集
#data_without_missing_values = original_data.dropna(axis=1)

##两个数据集
#cols_with_missing = [col for col in original_data.columns 
#                                 if original_data[col].isnull().any()]
#redued_original_data = original_data.drop(cols_with_missing, axis=1)
#reduced_test_data = test_data.drop(cols_with_missing, axis=1)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.imputation import Imputer

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

# Load data
melb_data = pd.read_csv('./melb_data.csv')

melb_target = melb_data.Price
melb_predictors = melb_data.drop(['Price'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])

X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors, 
                                                    melb_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

# drop columns with Missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))

# imputer
my_imputer = Imputer()
# 先fit再transform
# fit：只有X_train的话，执行无监督学习算法，比如降维、特征提取、标准化等
# transform：根据对象的特性来定，比如这里是Imputer()对象，那么就是要执行impute
# 另外也可以是StandardScaler()对象，实现标准化（在此之前也要fit）
#print(len(X_train.columns))
imputed_X_train = my_imputer.fit_transform(X_train) 
#print(len(imputed_X_train[0,:]))
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))

# 被impute的数据
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())

# 有缺失值得数据不是直接删除，而是有数据的是false，无数据的是true
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))


