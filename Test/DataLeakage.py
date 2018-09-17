# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 19:28:01 2018

数据泄露

（1）预测器泄露

比如预测一个人是否患有肺炎，有一个特征是是否吃抗生素。

吃抗生素-患肺炎，在数据中有很大的关联（没有患肺炎的都没有吃抗生素，而患肺炎的都吃抗生素）。
这是患肺炎后希望恢复而吃抗生素，造成了这种关系。但是吃抗生素是由患肺炎产生的。
这就是一个数据泄露。

    有用       |     无用
-------------------------------->时间
               |
            事件发生
            
如何避免？

* 寻找和目标有着高度相关性的列

* 如果发现模型有着高度的正确率，有可能出现了泄漏问题。

（2）验证策略泄露

比如在划分数据集和验证集前，对数据进行了处理。

因为验证集用于验证之前没有考虑过得数据是否正确，如果在数据划分前进行了处理，则没有了验证效果。

@author: HP528

"""

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

data = pd.read_csv('./AER_credit_card_data.csv', 
                   true_values = ['yes'],
                   false_values = ['no'])
print(data.head())

y = data.card
X = data.drop(['card'], axis=1)

# Since there was no preprocessing, we didn't need a pipeline here. Used anyway as best practice
modeling_pipeline = make_pipeline(RandomForestClassifier())
cv_scores = cross_val_score(modeling_pipeline, X, y, scoring='accuracy')
print("Cross-val accuracy: %f" %cv_scores.mean())

expenditures_cardholders = data.expenditure[data.card]
expenditures_noncardholders = data.expenditure[~data.card]


# 输出有卡没有支出和没有卡也没有支出的比例
print('Fraction of those who received a card with no expenditures: %.2f' \
      %(( expenditures_cardholders == 0).mean()))
print('Fraction of those who not received a card with no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean()))

potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)
cv_scores = cross_val_score(modeling_pipeline, X2, y, scoring='accuracy')
print("Cross-val accuracy: %f" %cv_scores.mean())

























