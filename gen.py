#!/usr/bin/env python
# -- coding:utf-8 --

import pandas as pd
import tools
import numpy as np
import os


def _second_min(x):
    tmp = np.sort(x.as_matrix())
    return None if len(tmp) < 2 else tmp[1]


def _third_min(x):
    tmp = np.sort(x.as_matrix())
    return None if len(tmp) < 3 else tmp[2]


def makeY(date_begin, folder):
    if date_begin == '2016-07-01':
        frame = pd.read_csv('data/product_market.csv')
        frame['y'] = 0
        frame.to_csv(folder + '/' + 'y.csv', index=False)
        return
    date_end = tools.move_day(date_begin, 30)
    frame = pd.read_csv('data/farming.csv')
    frame = frame[(frame.time >= date_begin) & (frame.time <= date_end)]
    frame[['province', 'market', 'type', 'name', 'time', 'avgprice']].rename(columns={'avgprice': 'y'}).to_csv(
        folder + '/' + 'y.csv', index=False)
    pass


# 特征V1
def makeV1(date_begin, folder):
    frame = pd.read_csv('data/farming.csv')
    frame_final = pd.read_csv(folder + '/' + 'y.csv')
    for day in [1, 2, 3, 4, 7, 14, 21, 30, 60]:
        date1 = tools.move_day(date_begin, -day)
        frame1 = frame[(frame.time >= date1) & (frame.time < date_begin)]
        frame1 = frame1.groupby(['province', 'market', 'type', 'name'], as_index=False)['avgprice'].agg(
            {'_' + str(day) + 'day_avg': np.mean,
             '_' + str(day) + 'day_std': np.std,
             '_' + str(day) + 'day_min': np.min,
             })
        frame_final = pd.merge(frame_final, frame1, how='left', on=['province', 'market', 'type', 'name'])
    frame_final.drop(['y'], axis=1, inplace=True)
    frame_final.to_csv(folder + '/' + 'v1.csv', index=False)
    pass


# 特征V2
def makeV2(date_begin, folder):
    frame = pd.read_csv('data/farming.csv')
    frame_final = pd.read_csv(folder + '/' + 'y.csv')
    frame.sort_values(by=['time'], inplace=True)
    item_group = frame.groupby(['province', 'market', 'type', 'name'], as_index=False)
    frame_final = pd.merge(frame_final,
                           item_group['time'].agg({
                               '_last_sell': (
                                   lambda x: -1 if len(x) == 0 else tools.time_diff(x.as_matrix()[-1], date_begin)),
                               '_first_sell': (
                                   lambda x: -1 if len(x) == 0 else tools.time_diff(x.as_matrix()[0], date_begin)),
                               # '_has_record': np.size,
                           }), how='left')
    frame_final = pd.merge(frame_final, item_group['avgprice'].agg({'_last_price': lambda x: x.as_matrix()[-1]}),
                           how='left')
    frame_final.drop(['y'], axis=1, inplace=True)
    frame_final.to_csv(folder + '/' + 'v2.csv', index=False)
    pass


# 特征V3
def makeV3(date_begin, folder):
    frame = pd.read_csv('data/farming.csv')
    frame.sort_values(by=['time'], inplace=True)
    frame_final = pd.read_csv(folder + '/' + 'y.csv')
    frame = frame[frame.time < date_begin]
    frame_group = frame.groupby(['province', 'market', 'type', 'name'], as_index=False)
    for day in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        frame1 = frame_group['avgprice'].agg(
            {'_' + str(day) + 'day_exists_avg': lambda x: np.mean(x.as_matrix()[-day:]),
             '_' + str(day) + 'day_exists_std': lambda x: np.std(x.as_matrix()[-day:]),
             '_' + str(day) + 'day_exists_min': lambda x: np.min(x.as_matrix()[-day:]),
             })
        frame_final = pd.merge(frame_final, frame1, how='left', on=['province', 'market', 'type', 'name'])
    frame_final.drop(['y'], axis=1, inplace=True)
    frame_final.to_csv(folder + '/' + 'v3.csv', index=False)
    pass


def makeV4(date_begin, folder):
    frame = pd.read_csv('data/farming.csv')
    frame_final = pd.read_csv(folder + '/' + 'y.csv')
    for day in [1, 2, 3, 4, 7, 14, 21, 30, 60]:
        date1 = tools.move_day(date_begin, -day)
        frame1 = frame[(frame.time >= date1) & (frame.time < date_begin)]
        frame1 = frame1.groupby(['province', 'market', 'type', 'name'], as_index=False)['avgprice'].agg(
            {'_' + str(day) + 'day_size': lambda x: len(x),
             })
        frame_final = pd.merge(frame_final, frame1, how='left', on=['province', 'market', 'type', 'name'])
    frame_final.drop(['y'], axis=1, inplace=True)
    frame_final.to_csv(folder + '/' + 'v4.csv', index=False)
    pass


def makeV5(date_begin, folder):
    frame = pd.read_csv('data/farming.csv')
    frame.sort_values(by=['time'], inplace=True)
    frame['minprice'] = frame['minprice'].apply(lambda x: None if x == 0.0 else x)
    frame_final = pd.read_csv(folder + '/' + 'y.csv')
    frame = frame[frame.time < date_begin]
    frame_group = frame.groupby(['province', 'market', 'type', 'name'], as_index=False)
    for day in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        frame1 = frame_group['minprice'].agg(
            {'_' + str(day) + 'day_exists_avg_min': lambda x: np.mean(x.as_matrix()[-day:]),
             '_' + str(day) + 'day_exists_std_min': lambda x: np.std(x.as_matrix()[-day:]),
             '_' + str(day) + 'day_exists_min_min': lambda x: np.min(x.as_matrix()[-day:]),
             })
        frame_final = pd.merge(frame_final, frame1, how='left', on=['province', 'market', 'type', 'name'])
    frame_final.drop(['y'], axis=1, inplace=True)
    frame_final.to_csv(folder + '/' + 'v5.csv', index=False)
    pass


def makeV6(date_begin, folder):
    frame = pd.read_csv('data/farming.csv')
    frame.sort_values(by=['time'], inplace=True)
    frame['maxprice'] = frame['maxprice'].apply(lambda x: None if x == 0.0 else x)
    frame_final = pd.read_csv(folder + '/' + 'y.csv')
    frame = frame[frame.time < date_begin]
    frame_group = frame.groupby(['province', 'market', 'type', 'name'], as_index=False)
    for day in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        frame1 = frame_group['maxprice'].agg(
            {'_' + str(day) + 'day_exists_avg_max': lambda x: np.mean(x.as_matrix()[-day:]),
             '_' + str(day) + 'day_exists_std_max': lambda x: np.std(x.as_matrix()[-day:]),
             '_' + str(day) + 'day_exists_min_max': lambda x: np.min(x.as_matrix()[-day:]),
             })
        frame_final = pd.merge(frame_final, frame1, how='left', on=['province', 'market', 'type', 'name'])
    frame_final.drop(['y'], axis=1, inplace=True)
    frame_final.to_csv(folder + '/' + 'v6.csv', index=False)
    pass


# 特征V7
def makeV7(date_begin, folder):
    frame = pd.read_csv('data/farming.csv')
    frame_final = pd.read_csv(folder + '/' + 'y.csv')
    for day in [1, 2, 3, 4, 7, 14, 21, 30, 60]:
        date1 = tools.move_day(date_begin, -day)
        frame1 = frame[(frame.time >= date1) & (frame.time < date_begin)]
        frame1 = frame1.groupby(['province', 'market', 'type', 'name'], as_index=False)['avgprice'].agg(
            {'_' + str(day) + 'day_offset': lambda x: np.max(x) - np.min(x),
             '_' + str(day) + 'day_min2': _second_min,
             '_' + str(day) + 'day_min3': _third_min,
             })
        frame_final = pd.merge(frame_final, frame1, how='left', on=['province', 'market', 'type', 'name'])
    frame_final.drop(['y'], axis=1, inplace=True)
    frame_final.to_csv(folder + '/' + 'v7.csv', index=False)
    pass


# 特征V8
def makeV8(date_begin, folder):
    frame = pd.read_csv('data/farming.csv')
    frame_final = pd.read_csv(folder + '/' + 'y.csv')
    for day in [1, 2, 3, 4, 7, 14, 21, 30, 60]:
        date1 = tools.move_day(date_begin, -day)
        frame1 = frame[(frame.time >= date1) & (frame.time < date_begin)]
        frame1 = frame1.groupby(['name', 'type'], as_index=False)['avgprice'].agg(
            {'_' + str(day) + 'day_avg': np.mean,
             })
        frame_final = pd.merge(frame_final, frame1, how='left', on=['type', 'name'])
    frame_final.drop(['y'], axis=1, inplace=True)
    frame_final.to_csv(folder + '/' + 'v8.csv', index=False)
    pass


# 特征V9
def makeV9(date_begin, folder):
    frame = pd.read_csv('data/farming.csv')
    frame_final = pd.read_csv(folder + '/' + 'y.csv')

    # 前半个月平均值
    date1 = tools.move_day(date_begin, -30)
    date2 = tools.move_day(date_begin, -15)
    frame1 = frame[(frame.time >= date1) & (frame.time < date2)]
    frame1 = frame1.groupby(['province', 'market', 'type', 'name'], as_index=False)['avgprice'].agg(
        {'_half_month_ago_day_avg': np.mean, })
    frame_final = pd.merge(frame_final, frame1, how='left', on=['province', 'market', 'type', 'name'])

    # 前1个月平均值
    date1 = tools.move_day(date_begin, -60)
    date2 = tools.move_day(date_begin, -30)
    frame1 = frame[(frame.time >= date1) & (frame.time < date2)]
    frame1 = frame1.groupby(['province', 'market', 'type', 'name'], as_index=False)['avgprice'].agg(
        {'_one_month_ago_day_avg': np.mean, })
    frame_final = pd.merge(frame_final, frame1, how='left', on=['province', 'market', 'type', 'name'])
    frame_final.drop(['y'], axis=1, inplace=True)
    frame_final.to_csv(folder + '/' + 'v9.csv', index=False)


# 特征V10
def makeV10(date_begin, folder):
    frame = pd.read_csv('data/farming.csv')
    frame_final = pd.read_csv(folder + '/' + 'y.csv')
    TYPES = frame['type'].drop_duplicates().as_matrix()
    for i in xrange(len(TYPES)):
        frame_final['_is_type' + str(i)] = frame_final['type'].apply(lambda x: 1 if x == TYPES[i] else 0)
    frame_final.drop(['y'], axis=1, inplace=True)
    frame_final.to_csv(folder + '/' + 'v10.csv', index=False)
    pass


# 特征V11
def makeV11(date_begin, folder):
    frame_final = pd.read_csv(folder + '/' + 'y.csv')
    frame_final['time_diff'] = frame_final['time'].apply(lambda x: tools.time_diff(x, date_begin))
    frame_final.drop(['y'], axis=1, inplace=True)
    frame_final.to_csv(folder + '/' + 'v11.csv', index=False)
    pass


# 特征V12
def makeV12(date_begin, folder):
    frame_final = pd.read_csv(folder + '/' + 'y.csv')
    frame_final['time_diff'] = frame_final['time'].apply(lambda x: np.log(tools.time_diff(x, date_begin)))
    frame_final.drop(['y'], axis=1, inplace=True)
    frame_final.to_csv(folder + '/' + 'v12.csv', index=False)
    pass


# 特征V13
def makeV13(date_begin, folder):
    frame_final = pd.read_csv(folder + '/' + 'v1.csv')
    DAYS = [60, 30, 21, 14, 7, 4, 3, 2, 1]
    for i in xrange(len(DAYS) - 1):
        frame_final['_' + str(DAYS[i + 1]) + 'day_avg'] = frame_final.apply(
            lambda x: x['_' + str(DAYS[i]) + 'day_avg'] if np.isnan(x['_' + str(DAYS[i + 1]) + 'day_avg']) else
            x['_' + str(DAYS[i + 1]) + 'day_avg'], axis=1
        )
        frame_final['_' + str(DAYS[i + 1]) + 'day_min'] = frame_final.apply(
            lambda x: x['_' + str(DAYS[i]) + 'day_min'] if np.isnan(x['_' + str(DAYS[i + 1]) + 'day_min']) else
            x['_' + str(DAYS[i + 1]) + 'day_avg'], axis=1
        )
    frame_final.to_csv(folder + '/' + 'v13.csv', index=False)
    pass

DATES = ['2016-06-01', '2016-05-25', '2016-05-01', '2016-04-25', '2016-07-01', '2015-06-01', '2015-07-01',
         '2016-04-01', '2016-03-01', '2016-05-15', '2016-05-20', '2016-04-15', '2016-04-20']
FOLDER = []

for i in xrange(len(DATES)):
    FOLDER.append('dataset/' + DATES[i])
    if not os.path.exists(FOLDER[i]):
        os.mkdir(FOLDER[i])
    makeY(DATES[i], FOLDER[i])
    makeV1(DATES[i], FOLDER[i])
    makeV2(DATES[i], FOLDER[i])
    makeV3(DATES[i], FOLDER[i])
    # makeV4(DATES[i], FOLDER[i])
    # makeV5(DATES[i], FOLDER[i])
    # makeV6(DATES[i], FOLDER[i])
    # makeV7(DATES[i], FOLDER[i])
    # makeV8(DATES[i], FOLDER[i])
    # makeV9(DATES[i], FOLDER[i])
    # makeV10(DATES[i], FOLDER[i])
    makeV11(DATES[i], FOLDER[i])
    # makeV12(DATES[i], FOLDER[i])
    # makeV13(DATES[i], FOLDER[i])
