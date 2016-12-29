#!/usr/bin/env python
# -- coding:utf-8 --

import datetime
import math
import pandas as pd
import time
import MySQLdb
import os


def date_range(begin, end, time_regex='%Y-%m-%d'):
    '''
        生成begin到end的每一天的一个list
    :param
        begin: str 开始时间
        end: str 结束时间
        time_regex: str 时间格式的正则表达式
    :argument
        begin需要小于等于end
    :return:
        day_range: list
    --------
        如 date_range('20151220', '20151223')返回 ['20151220', '20151221', '20151222', '20151223']
    '''
    day_range = []
    day = datetime.datetime.strptime(begin, time_regex).date()
    while True:
        day_str = datetime.datetime.strftime(day, time_regex)
        day_range.append(day_str)
        if day_str == end:
            break
        day = day + datetime.timedelta(days=1)
    return day_range


def move_day(day_str, offset, time_regex='%Y-%m-%d'):
    '''
        计算day_str偏移offset天后的日期
    :param
        day_str: str 原时间
        offset: str 要偏移的天数
        time_regex: str 时间字符串的正则式
    :return:
        day_str: str 运算之后的结果时间, 同样以time_regex的格式返回
    --------
        如 move_day('20151228', 1)返回 '20151229'
    '''
    day = datetime.datetime.strptime(day_str, time_regex).date()
    day = day + datetime.timedelta(days=offset)
    day_str = datetime.datetime.strftime(day, time_regex)
    return day_str


def move_hours(day_str, offset, time_regex='%Y/%m/%d %H:%M:%S'):
    '''
        计算day_str偏移offset天后的日期
    :param
        day_str: str 原时间
        offset: str 要偏移的天数
        time_regex: str 时间字符串的正则式
    :return:
        day_str: str 运算之后的结果时间, 同样以time_regex的格式返回
    --------
        如 move_day('20151228', 1)返回 '20151229'
    '''
    t = datetime.datetime.strptime(day_str, time_regex)
    t = t + datetime.timedelta(hours=offset)
    day_str = datetime.datetime.strftime(t, time_regex)
    return day_str


def time_diff(day_str1, day_str2, time_regex='%Y-%m-%d'):
    '''
        计算day_str1和day_str2的日期差
    '''
    day_str1 = str(day_str1)
    day_str2 = str(day_str2)
    day1 = datetime.datetime.strptime(day_str1, time_regex).date()
    day2 = datetime.datetime.strptime(day_str2, time_regex).date()
    return math.fabs((day1 - day2).days)


def str2time_stamp(_str, re='%Y%m%d %H'):
    return int(time.mktime(time.strptime(_str, re)))


def time_stamp2str(stamp, re='%Y%m%d %H:%M:%S'):
    return time.strftime(re, time.localtime(stamp))


def cross_join(frame1, frame2):
    '''
        笛卡尔积
    :param frame1:
    :param frame2:
    :return:
    '''
    frame1['_tmpkey'] = 0
    frame2['_tmpkey'] = 0
    frame = pd.merge(frame1, frame2, how='outer', on='_tmpkey')
    frame.drop('_tmpkey', axis=1, inplace=True)
    frame1.drop('_tmpkey', axis=1, inplace=True)
    frame2.drop('_tmpkey', axis=1, inplace=True)
    return frame


def merge_table(artist_frame, date_frame, Y_frame):
    frame1 = pd.merge(Y_frame, date_frame, how='left', on=['date', 'artist_id'])
    return pd.merge(frame1, artist_frame, how='left', on=['artist_id'])


def get_week(date, re='%Y-%m-%d'):
    day = datetime.datetime.strptime(str(date), re).date()
    return int(day.strftime("%w"))


__BEFORE_HOLIDAY_WEIGHT = {
    '20150403': 1, '20150430': 1, '20150619': 1,
    '20150925': 1,

}
__HOLIDAY_WEIGHT = {
    '20150404': 1, '20150405': 1, '20150406': 1,
    '20150501': 1, '20150502': 1, '20150503': 1,
    '20150620': 1, '20150621': 1, '20150622': 1,
    '20150926': 1, '20150927': 1, '20150928': 1,
    '20150929': 1, '20150930': 1, '20151001': 1,
    '20151002': 1, '20151003': 1, '20151004': 1,
    '20151005': 1, '20151006': 1, '20151007': 1,
}


def get_holiday_weight(date):
    if str(date) in __HOLIDAY_WEIGHT:
        return __HOLIDAY_WEIGHT[str(date)]
    return 0


def get_before_holiday_weight(date):
    if str(date) in __BEFORE_HOLIDAY_WEIGHT:
        return __BEFORE_HOLIDAY_WEIGHT[str(date)]
    return 0
