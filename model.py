#!/usr/bin/env python
# -- coding:utf-8 --

from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
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
    train_set = pd.concat(
        [dfs(0, len(feature_files), feature_files + ['y'], 'dataset/' + date) for date in training_dates])
    test_set = dfs(0, len(feature_files), feature_files + ['y'], 'dataset/2016-06-01')
    test1_set = dfs(0, len(feature_files), feature_files + ['y'], 'dataset/2016-05-25')
    # train_set.to_csv('train_set.csv', index=False)
    # test_set.to_csv('test_set.csv', index=False)

    '''
    unique_size = pd.read_csv('unique_size.csv')
    train_set = pd.merge(train_set, unique_size, how='left')
    train_set = train_set[train_set.unique_size > 1]
    train_set.drop(['unique_size'], axis=1, inplace=True)
    '''

    train_set = train_set.fillna(-1, downcast='infer')
    test_set = test_set.fillna(-1, downcast='infer')
    test1_set = test1_set.fillna(-1, downcast='infer')

    train_set['y_log'] = train_set['y'].apply(lambda x: np.log(1 + x))
    test_set['y_log'] = test_set['y'].apply(lambda x: np.log(1 + x))
    test1_set['y_log'] = test1_set['y'].apply(lambda x: np.log(1 + x))

    feature_set = filter(lambda x: x not in ['y', 'time', 'province', 'market', 'name', 'type', 'y_log'],
                         train_set.columns)

    scaler = StandardScaler()
    scaler.fit(train_set[feature_set].as_matrix())

    # model1
    model1 = LinearRegression(normalize=True)
    model1.fit(scaler.transform(train_set[feature_set].as_matrix()), train_set['y'].as_matrix(),
               sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
               )
    print zip(feature_set, model1.coef_)
    test_set['predictY'] = model1.predict(scaler.transform(test_set[feature_set].as_matrix()))
    test_set.to_csv('result/' + feature_set_folder + '/model1_offline.csv')
    test1_set['predictY'] = model1.predict(scaler.transform(test1_set[feature_set].as_matrix()))
    test1_set.to_csv('result/' + feature_set_folder + '/model1_offline1.csv')

    # model2
    model2 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=6, colsample_bytree=0.7, subsample=0.7,
                          colsample_bylevel=0.7)
    model2.fit(train_set[feature_set].as_matrix(), train_set['y'].as_matrix(),
               sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
               )
    test_set['predictY'] = model2.predict(test_set[feature_set].as_matrix())
    test_set.to_csv('result/' + feature_set_folder + '/model2_offline.csv')
    test1_set['predictY'] = model2.predict(test1_set[feature_set].as_matrix())
    test1_set.to_csv('result/' + feature_set_folder + '/model2_offline1.csv')

    # model3
    model3 = LinearSVR(tol=1e-7)
    model3.fit(scaler.transform(train_set[feature_set].as_matrix()), train_set['y'].as_matrix(),
               sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
               )
    test_set['predictY'] = model3.predict(scaler.transform(test_set[feature_set].as_matrix()))
    test_set.to_csv('result/' + feature_set_folder + '/model3_offline.csv')
    test1_set['predictY'] = model3.predict(scaler.transform(test1_set[feature_set].as_matrix()))
    test1_set.to_csv('result/' + feature_set_folder + '/model3_offline1.csv')

    # model4
    model4 = RandomForestRegressor(n_estimators=1000, max_depth=7, max_features=0.2, max_leaf_nodes=100)
    model4.fit(train_set[feature_set].as_matrix(), train_set['y'].as_matrix(),
               sample_weight=np.array(map(lambda x: 1.0 / x / x, train_set['y'].as_matrix()))
               )
    test_set['predictY'] = model4.predict(test_set[feature_set].as_matrix())
    test_set.to_csv('result/' + feature_set_folder + '/model4_offline.csv')
    test1_set['predictY'] = model4.predict(test1_set[feature_set].as_matrix())
    test1_set.to_csv('result/' + feature_set_folder + '/model4_offline1.csv')

    # model15
    model15 = ExtraTreesRegressor(n_estimators=1000, max_depth=12, max_features=0.3, max_leaf_nodes=400)
    model15.fit(train_set[feature_set].as_matrix(), train_set['y'].as_matrix(),
                sample_weight=np.array(map(lambda x: 1.0 / x / x, train_set['y'].as_matrix()))
                )
    test_set['predictY'] = model15.predict(test_set[feature_set].as_matrix())
    test_set.to_csv('result/' + feature_set_folder + '/model15_offline.csv')
    test1_set['predictY'] = model15.predict(test1_set[feature_set].as_matrix())
    test1_set.to_csv('result/' + feature_set_folder + '/model15_offline1.csv')

    # model5
    model5 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=6, colsample_bytree=0.7, subsample=0.7,
                          colsample_bylevel=0.7, seed=10000)
    model5.fit(train_set[feature_set].as_matrix(), train_set['y'].as_matrix(),
               sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
               )
    test_set['predictY'] = model5.predict(test_set[feature_set].as_matrix())
    test_set.to_csv('result/' + feature_set_folder + '/model5_offline.csv')
    test1_set['predictY'] = model5.predict(test1_set[feature_set].as_matrix())
    test1_set.to_csv('result/' + feature_set_folder + '/model5_offline1.csv')

    # model6
    model6 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=5, colsample_bytree=0.7, subsample=0.7,
                          colsample_bylevel=0.7)
    model6.fit(train_set[feature_set].as_matrix(), train_set['y'].as_matrix(),
               sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
               )
    test_set['predictY'] = model6.predict(test_set[feature_set].as_matrix())
    test_set.to_csv('result/' + feature_set_folder + '/model6_offline.csv')
    test1_set['predictY'] = model6.predict(test1_set[feature_set].as_matrix())
    test1_set.to_csv('result/' + feature_set_folder + '/model6_offline1.csv')

    pass


def submit(feature_files, training_dates, feature_set_folder):
    train_set = pd.concat(
        [dfs(0, len(feature_files), feature_files + ['y'], 'dataset/' + date) for date in training_dates])
    test_set = dfs(0, len(feature_files), feature_files + ['y'], 'dataset/2016-07-01')

    '''
    unique_size = pd.read_csv('unique_size.csv')
    train_set = pd.merge(train_set, unique_size, how='left')
    train_set = train_set[train_set.unique_size > 1]
    train_set.drop(['unique_size'], axis=1, inplace=True)
    '''

    train_set = train_set.fillna(-1, downcast='infer')
    test_set = test_set.fillna(-1, downcast='infer')

    train_set['y_log'] = train_set['y'].apply(lambda x: np.log(1 + x))
    test_set['y_log'] = test_set['y'].apply(lambda x: np.log(1 + x))

    feature_set = filter(
        lambda x: x not in ['y', 'time', 'province', 'market', 'name', 'type', 'y_log'],
        train_set.columns)

    scaler = StandardScaler()
    scaler.fit(train_set[feature_set].as_matrix())

    # model1
    model1 = LinearRegression(normalize=True)
    model1.fit(scaler.transform(train_set[feature_set].as_matrix()), train_set['y'].as_matrix(),
               sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
               )
    print zip(feature_set, model1.coef_)
    test_set['predictY'] = model1.predict(scaler.transform(test_set[feature_set].as_matrix()))
    test_set.to_csv('result/' + feature_set_folder + '/model1_online.csv')

    # model2
    model2 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=6, colsample_bytree=0.7, subsample=0.7,
                          colsample_bylevel=0.7)
    model2.fit(train_set[feature_set].as_matrix(), train_set['y'].as_matrix(),
               sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
               )
    test_set['predictY'] = model2.predict(test_set[feature_set].as_matrix())
    test_set.to_csv('result/' + feature_set_folder + '/model2_online.csv')

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
    test_set.to_csv('result/' + feature_set_folder + '/model4_online.csv')

    # model5
    model5 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=6, colsample_bytree=0.7, subsample=0.7,
                          colsample_bylevel=0.7, seed=10000)
    model5.fit(train_set[feature_set].as_matrix(), train_set['y'].as_matrix(),
               sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
               )
    test_set['predictY'] = model5.predict(test_set[feature_set].as_matrix())
    test_set.to_csv('result/' + feature_set_folder + '/model5_online.csv')

    # model6
    model6 = XGBRegressor(n_estimators=600, learning_rate=0.01, max_depth=5, colsample_bytree=0.7, subsample=0.7,
                          colsample_bylevel=0.7)
    model6.fit(train_set[feature_set].as_matrix(), train_set['y'].as_matrix(),
               sample_weight=map(lambda x: 1.0 / x / x, train_set['y'].as_matrix())
               )
    test_set['predictY'] = model6.predict(test_set[feature_set].as_matrix())
    test_set.to_csv('result/' + feature_set_folder + '/model6_online.csv')

    pass


run(['v1', 'v3', 'v11'], ['2016-05-01', '2016-04-25'], 'feature_set1')
# run(['v1', 'v2','v3', 'v11'], ['2016-05-01', '2016-04-25'], 'feature_set2')
# submit(['v1', 'v3', 'v11'], ['2016-06-01', '2016-05-25'], 'feature_set1')
submit(['v1', 'v2', 'v3', 'v11'], ['2016-06-01', '2016-05-25'], 'feature_set2')
