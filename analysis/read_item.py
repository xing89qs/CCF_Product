#!/usr/bin/env python
# -- coding:utf-8 --

import pandas as pd
import matplotlib.pyplot as plt

NAME = 'F95E90C9764A4FCCA8FA3648DBCDBE1B'
MARKET = 'F84FFE619392149018384D16BE6FF525'
TYPE = '配花类'

frame = pd.read_csv('../data/farming.csv')
tmp = \
    frame[(frame.name == NAME) & (frame.market == MARKET) & (
        frame.type == TYPE)][
        ['avgprice', 'time']]
tmp = tmp.sort_values(by=['time'])
for index, row in tmp.iterrows():
    print row['time'], row['avgprice']

plt.plot(xrange(len(tmp['time'])), tmp['avgprice'], 'b')
plt.show()

'''
len1 = len(tmp)

frame = pd.read_csv('../current.csv')
tmp = \
    frame[(frame.name == NAME) & (frame.market == MARKET) & (
        frame.type == TYPE)][
        ['predictY', 'time']]
plt.plot(xrange(len1 - len(tmp), len1), tmp['predictY'], 'g')

frame = pd.read_csv('../submit_12_13_2.csv', header=None)
frame.columns = ['market', 'type', 'name', 'time', 'predictY']
tmp = \
    frame[(frame.name == NAME) & (frame.market == MARKET) & (
        frame.type == TYPE)][
        ['predictY', 'time']]

print len1, len1 + len(tmp)
plt.plot(xrange(len1, len1 + len(tmp), 1), tmp['predictY'], 'r')
'''