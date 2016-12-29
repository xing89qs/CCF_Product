#!/usr/bin/env python
# -- coding:utf-8 --

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def calculate(test_set):
    return np.mean(
        test_set[test_set.y != -1].apply(
            lambda x: ((x['predictY'] - x['y']) / x['y']) ** 2,
            axis=1))


def merge(x):
    if x['unique_size'] == 1 and x['_1day_exists_avg'] != -1:
        return x['_1day_exists_avg']
    if x['predictY'] < 0:
        return x['predictY2'] * 0.5 + 0.5 * x['predictY4']
    # 线性回归值很大直接用
    if x['predictY1'] > 100:
        return x['predictY1']

    return x['predictY']


f1 = pd.read_csv('result/feature_set9/model1_online.csv')
f1 = f1.rename(columns={'predictY': 'predictY1'})
f2 = pd.read_csv('result/feature_set9/model2_online.csv')
f2 = f2.rename(columns={'predictY': 'predictY2'})
f4 = pd.read_csv('result/feature_set9/model4_online.csv')
f4 = f4.rename(columns={'predictY': 'predictY4'})
f5 = pd.read_csv('result/feature_set9/model5_online.csv')
f5 = f5.rename(columns={'predictY': 'predictY5'})
f6 = pd.read_csv('result/feature_set9/model6_online.csv')
f6 = f6.rename(columns={'predictY': 'predictY6'})
f1_1 = pd.read_csv('result/feature_set1/model1_online.csv')
f1_1 = f1_1.rename(columns={'predictY': 'predictY1_1'})
f2_1 = pd.read_csv('result/feature_set1/model2_online.csv')
f2_1 = f2_1.rename(columns={'predictY': 'predictY2_1'})
f4_1 = pd.read_csv('result/feature_set1/model4_online.csv')
f4_1 = f4_1.rename(columns={'predictY': 'predictY4_1'})

print len(f1), len(f2), len(f4), len(f5), len(f6), len(f1_1), len(f2_1), len(f4_1)

new_test = f1
new_test = pd.merge(new_test, f2, how='left')
new_test = pd.merge(new_test, f4, how='left')
new_test = pd.merge(new_test, f5, how='left')
new_test = pd.merge(new_test, f6, how='left')
new_test = pd.merge(new_test, f1_1, how='left')
new_test = pd.merge(new_test, f2_1, how='left')
new_test = pd.merge(new_test, f4_1, how='left')

f1 = pd.read_csv('result/feature_set9/model1_online_stacking1.csv')
f1 = f1.rename(columns={'predictY': 'predictY1'})
f2 = pd.read_csv('result/feature_set9/model2_online_stacking1.csv')
f2 = f2.rename(columns={'predictY': 'predictY2'})
f4 = pd.read_csv('result/feature_set9/model4_online_stacking1.csv')
f4 = f4.rename(columns={'predictY': 'predictY4'})
f5 = pd.read_csv('result/feature_set9/model5_online_stacking1.csv')
f5 = f5.rename(columns={'predictY': 'predictY5'})
f6 = pd.read_csv('result/feature_set9/model6_online_stacking1.csv')
f6 = f6.rename(columns={'predictY': 'predictY6'})
f1_1 = pd.read_csv('result/feature_set1/model1_online_stacking1.csv')
f1_1 = f1_1.rename(columns={'predictY': 'predictY1_1'})
f2_1 = pd.read_csv('result/feature_set1/model2_online_stacking1.csv')
f2_1 = f2_1.rename(columns={'predictY': 'predictY2_1'})
f4_1 = pd.read_csv('result/feature_set1/model4_online_stacking1.csv')
f4_1 = f4_1.rename(columns={'predictY': 'predictY4_1'})

new_train1 = f1
new_train1 = pd.merge(new_train1, f1, how='left')
new_train1 = pd.merge(new_train1, f2, how='left')
new_train1 = pd.merge(new_train1, f4, how='left')
new_train1 = pd.merge(new_train1, f5, how='left')
new_train1 = pd.merge(new_train1, f6, how='left')
new_train1 = pd.merge(new_train1, f1_1, how='left')
new_train1 = pd.merge(new_train1, f2_1, how='left')
new_train1 = pd.merge(new_train1, f4_1, how='left')
print len(f1), len(new_train1)

f1 = pd.read_csv('result/feature_set9/model1_online_stacking2.csv')
f1 = f1.rename(columns={'predictY': 'predictY1'})
f2 = pd.read_csv('result/feature_set9/model2_online_stacking2.csv')
f2 = f2.rename(columns={'predictY': 'predictY2'})
f4 = pd.read_csv('result/feature_set9/model4_online_stacking2.csv')
f4 = f4.rename(columns={'predictY': 'predictY4'})
f5 = pd.read_csv('result/feature_set9/model5_online_stacking2.csv')
f5 = f5.rename(columns={'predictY': 'predictY5'})
f6 = pd.read_csv('result/feature_set9/model6_online_stacking2.csv')
f6 = f6.rename(columns={'predictY': 'predictY6'})
f1_1 = pd.read_csv('result/feature_set1/model1_online_stacking2.csv')
f1_1 = f1_1.rename(columns={'predictY': 'predictY1_1'})
f2_1 = pd.read_csv('result/feature_set1/model2_online_stacking2.csv')
f2_1 = f2_1.rename(columns={'predictY': 'predictY2_1'})
f4_1 = pd.read_csv('result/feature_set1/model4_online_stacking2.csv')
f4_1 = f4_1.rename(columns={'predictY': 'predictY4_1'})

new_train2 = f1
new_train2 = pd.merge(new_train2, f1, how='left')
new_train2 = pd.merge(new_train2, f2, how='left')
new_train2 = pd.merge(new_train2, f4, how='left')
new_train2 = pd.merge(new_train2, f5, how='left')
new_train2 = pd.merge(new_train2, f6, how='left')
new_train2 = pd.merge(new_train2, f1_1, how='left')
new_train2 = pd.merge(new_train2, f2_1, how='left')
new_train2 = pd.merge(new_train2, f4_1, how='left')

print len(f1), len(new_train2)

new_train = pd.concat([new_train1, new_train2])

model1 = LinearRegression(normalize=True)
feature_set = ['predictY2', 'predictY4', '_last_price_all']
model1.fit(new_train[feature_set].as_matrix(), new_train['y'].as_matrix(),
           sample_weight=map(lambda x: 1.0 / x / x, new_train['y'].as_matrix())
           )

print model1.coef_

new_test['predictY'] = new_test['predictY2_1']
print len(new_test)
# print calculate(new_test)

new_test['predictY'] = model1.predict(new_test[feature_set].as_matrix())
unique_size = pd.read_csv('unique_size.csv')
new_test = pd.merge(new_test, unique_size, how='left')
new_test['predictY'] = new_test.apply(merge, axis=1)

print new_test['_1day_exists_avg'].min()

new_test[['market', 'type', 'name', 'time', 'predictY']].to_csv('submit_12_15_2.csv', header=None, index=False)

print calculate(new_test)
