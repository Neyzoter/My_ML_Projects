# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 16:46:19 2018

Pipeline

@author: HP528
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error

# Read Data
data = pd.read_csv('../input/melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price
train_X, test_X, train_y, test_y = train_test_split(X, y)

# pipeline
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())

# fit pipeline
my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)

######  和pipeline相同效果
my_imputer = Imputer()
my_model = RandomForestRegressor()

imputed_train_X = my_imputer.fit_transform(train_X)
imputed_test_X = my_imputer.transform(test_X)
my_model.fit(imputed_train_X, train_y)
predictions = my_model.predict(imputed_test_X)

