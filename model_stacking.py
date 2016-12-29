#!/usr/bin/env python
# -- coding:utf-8 --

from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


def dfs(l, r, feature_file, folder):
    if l == r:
        return pd.read_csv(folder + '/' + feature_file[l] + '.csv')
    mid = (l + r) / 2
    l_frame = dfs(l, mid, feature_file, folder)
    r_frame = dfs(mid + 1, r, feature_file, folder)
    return pd.merge(l_frame, r_frame, how='left')


def calculate(test_set):
    return np.mean(
        test_set[test_set.y != -1].apply(
            lambda x: ((x['predictY'] - x['y']) / x['y']) ** 2,
            axis=1))


def diff1(x):
    if x['_3day_exists_avg'] is None:
        return None
    return x['_3day_exists_avg'] - x['_7day_exists_avg']


def diff2(x):
    if x['_7day_avg'] is None:
        return None
    return x['_7day_avg'] - x['_30day_avg']


def run(feature_files, training_dates, feature_set_folder):
    train_set1 = pd.concat(
        [dfs(0, len(feature_files), feature_files + ['y'], 'dataset/' + date) for date in training_dates])

    train_set = train_set1[train_set1.time_diff > 15]
    test_set = train_set1[train_set1.time_diff <= 15]

    train_set = train_set.fillna(-1, downcast='infer')
    test_set = test_set.fillna(-1, downcast='infer')

    train_set['y_log'] = train_set['y'].apply(lambda x: np.log(1 + x))
    test_set['y_log'] = test_set['y'].apply(lambda x: np.log(1 + x))

    feature_set = filter(lambda x: x not in ['y', 'time', 'province', 'market', 'name', 'type', 'y_log'],
                         train_set.columns)

    scaler = StandardScaler()
    scaler.fit(train_set[feature_set].as_matrix())
    #
    # # model1
    # model1 = LinearRegression(normalize=True)
    # model1.fit(scaler.transform(train_set[feature_set].as_matrix()), train_set['y'].as_matrix(),
    #            sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
    #            )
    # print zip(feature_set, model1.coef_)
    # test_set['predictY'] = model1.predict(scaler.transform(test_set[feature_set].as_matrix()))
    # test_set.to_csv('result/' + feature_set_folder + '/model1_offline_stacking1.csv')
    #
    # # model2
    # model2 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=6, colsample_bytree=0.7, subsample=0.7,
    #                       colsample_bylevel=0.7)
    # model2.fit(train_set[feature_set].as_matrix(), train_set['y'].as_matrix(),
    #            sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
    #            )
    # test_set['predictY'] = model2.predict(test_set[feature_set].as_matrix())
    # test_set.to_csv('result/' + feature_set_folder + '/model2_offline_stacking1.csv')
    #
    # # model3
    # model3 = LinearSVR(tol=1e-7)
    # model3.fit(scaler.transform(train_set[feature_set].as_matrix()), train_set['y'].as_matrix(),
    #            sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
    #            )
    # test_set['predictY'] = model3.predict(scaler.transform(test_set[feature_set].as_matrix()))
    # test_set.to_csv('result/' + feature_set_folder + '/model3_offline.csv')

    # model4
    model4 = RandomForestRegressor(n_estimators=1000, max_depth=7, max_features=0.2, max_leaf_nodes=100)
    model4.fit(train_set[feature_set].as_matrix(), train_set['y'].as_matrix(),
               sample_weight=np.array(map(lambda x: 1.0 / x / x, train_set['y'].as_matrix()))
               )
    test_set['predictY'] = model4.predict(test_set[feature_set].as_matrix())
    test_set.to_csv('result/' + feature_set_folder + '/model4_offline_stacking1.csv')

    # model5
    model5 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=6, colsample_bytree=0.7, subsample=0.7,
                          colsample_bylevel=0.7, seed=10000)
    model5.fit(train_set[feature_set].as_matrix(), train_set['y'].as_matrix(),
               sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
               )
    test_set['predictY'] = model5.predict(test_set[feature_set].as_matrix())
    test_set.to_csv('result/' + feature_set_folder + '/model5_offline_stacking1.csv')

    # model5
    model6 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=5, colsample_bytree=0.7, subsample=0.7,
                          colsample_bylevel=0.7)
    model6.fit(train_set[feature_set].as_matrix(), train_set['y'].as_matrix(),
               sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
               )
    test_set['predictY'] = model6.predict(test_set[feature_set].as_matrix())
    test_set.to_csv('result/' + feature_set_folder + '/model6_offline_stacking1.csv')

    pass


def submit(feature_files, training_dates, feature_set_folder):
    train_set1 = pd.concat(
        [dfs(0, len(feature_files), feature_files + ['y'], 'dataset/' + date) for date in training_dates])

    train_set = train_set1[train_set1.time_diff > 15]
    test_set = train_set1[train_set1.time_diff <= 15]

    train_set = train_set.fillna(-1, downcast='infer')
    test_set = test_set.fillna(-1, downcast='infer')

    train_set['y_log'] = train_set['y'].apply(lambda x: np.log(1 + x))
    test_set['y_log'] = test_set['y'].apply(lambda x: np.log(1 + x))

    feature_set = filter(lambda x: x not in ['y', 'time', 'province', 'market', 'name', 'type', 'y_log'],
                         train_set.columns)

    scaler = StandardScaler()
    scaler.fit(train_set[feature_set].as_matrix())
    #
    # model1
    model1 = LinearRegression(normalize=True)
    model1.fit(scaler.transform(train_set[feature_set].as_matrix()), train_set['y'].as_matrix(),
               sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
               )
    print zip(feature_set, model1.coef_)
    test_set['predictY'] = model1.predict(scaler.transform(test_set[feature_set].as_matrix()))
    test_set.to_csv('result/' + feature_set_folder + '/model1_online_stacking1.csv')
    print test_set

    # model2
    model2 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=6, colsample_bytree=0.7, subsample=0.7,
                          colsample_bylevel=0.7)
    model2.fit(train_set[feature_set].as_matrix(), train_set['y'].as_matrix(),
               sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
               )
    test_set['predictY'] = model2.predict(test_set[feature_set].as_matrix())
    test_set.to_csv('result/' + feature_set_folder + '/model2_online_stacking1.csv')

    # model3
    model3 = LinearSVR(tol=1e-7)
    model3.fit(scaler.transform(train_set[feature_set].as_matrix()), train_set['y'].as_matrix(),
               sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
               )
    test_set['predictY'] = model3.predict(scaler.transform(test_set[feature_set].as_matrix()))
    test_set.to_csv('result/' + feature_set_folder + '/model3_offline.csv')

    # model4
    model4 = RandomForestRegressor(n_estimators=1000, max_depth=7, max_features=0.2, max_leaf_nodes=100)
    model4.fit(train_set[feature_set].as_matrix(), train_set['y'].as_matrix(),
               sample_weight=np.array(map(lambda x: 1.0 / x / x, train_set['y'].as_matrix()))
               )
    test_set['predictY'] = model4.predict(test_set[feature_set].as_matrix())
    test_set.to_csv('result/' + feature_set_folder + '/model4_online_stacking1.csv')

    # model5
    model5 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=6, colsample_bytree=0.7, subsample=0.7,
                          colsample_bylevel=0.7, seed=10000)
    model5.fit(train_set[feature_set].as_matrix(), train_set['y'].as_matrix(),
               sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
               )
    test_set['predictY'] = model5.predict(test_set[feature_set].as_matrix())
    test_set.to_csv('result/' + feature_set_folder + '/model5_online_stacking1.csv')

    # model6
    model6 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=5, colsample_bytree=0.7, subsample=0.7,
                          colsample_bylevel=0.7)
    model6.fit(train_set[feature_set].as_matrix(), train_set['y'].as_matrix(),
               sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
               )
    test_set['predictY'] = model6.predict(test_set[feature_set].as_matrix())
    test_set.to_csv('result/' + feature_set_folder + '/model6_online_stacking1.csv')

    pass


# run(['v1', 'v3', 'v11'], ['2016-05-01', '2016-04-25'], 'feature_set1')
# run(['v1', 'v3', 'v11', 'v14'], ['2016-05-01', '2016-04-25'], 'feature_set9')
# submit(['v1', 'v3', 'v11', 'v14'], ['2016-06-01', '2016-05-25'], 'feature_set9')
# run(['v1', 'v2', 'v3', 'v11'], ['2016-05-01', '2016-04-25'], 'feature_set4')
# run(['v1', 'v2', 'v3', 'xxv11'], ['2016-05-01', '2016-04-25', '2016-04-20', '2016-04-15'], 'feature_set6')
# run(['v1', 'v2', 'v3', 'v10'], ['2016-05-01', '2016-04-25'], 'feature_set3')
# submit(['v1', 'v14', 'v3', 'v11'], ['2016-06-01', '2016-05-25'], 'feature_set9')
submit(['v1', 'v3', 'v11'], ['2016-06-01', '2016-05-25'], 'feature_set1')
# submit(['v1', 'v2', 'v3', 'v11', 'v14'], ['2016-06-01', '2016-05-25'], 'feature_set6')


f = pd.read_csv('result/feature_set1/model1_offline.csv')
print calculate(f)
