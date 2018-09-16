# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 14:49:07 2018

Partial Dependence Plots

@author: HP528
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.preprocessing import Imputer

cols_to_use = ['Distance', 'Landsize', 'BuildingArea']

def get_some_data():
    data = pd.read_csv('./melb_data.csv')
    y = data.Price
    X = data[cols_to_use]
    my_imputer = Imputer()
    imputed_X = my_imputer.fit_transform(X)
    return imputed_X, y
    

X, y = get_some_data()
my_model = GradientBoostingRegressor()
my_model.fit(X, y)
my_plots = plot_partial_dependence(my_model, 
                                   # 选取的特征的列数（第0和2列），超过2或者3就会变得比较难看
                                   features=[0,2],   
                                   X=X, 
                                   feature_names=cols_to_use, 
                                   grid_resolution=50)# 在等间距点上的数量，越大就有越多的噪声