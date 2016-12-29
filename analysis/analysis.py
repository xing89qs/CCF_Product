#!/usr/bin/env python
# -- coding:utf-8 --

import pandas as pd
import numpy as np
import tools

frame = pd.read_csv('../dataset/2016-07-01/v2.csv')
frame = frame[['name', 'type', 'province', 'market', '_first_sell']].drop_duplicates()
print len(frame[frame._first_sell > 500]), len(frame)

# frame = pd.read_csv('../data/farming.csv')
# f1 = frame[['name', 'type', 'market', 'province']].drop_duplicates()
# print f1.groupby(['market'], as_index=False)['name'].agg(len)

frame = pd.read_csv('../current.csv')
frame['diff'] = ((frame['y'] - frame['predictY']) / frame['y']) ** 2
frame.sort_values(by=['diff'], inplace=True, ascending=False)
frame = frame[0:5000]
print frame.groupby(['name', 'market', 'type'], as_index=False)['diff'].agg(
    {'size': np.size, 'avg': np.mean}).sort_values(
    by=['size'], ascending=False)
