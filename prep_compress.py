#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.integrate import trapz # 用于计算积分
from scipy import interpolate



inte_features = ['USAGE_OF_BACKING_FILM', 'USAGE_OF_DRESSER',
       'USAGE_OF_POLISHING_TABLE', 'USAGE_OF_DRESSER_TABLE',
       'PRESSURIZED_CHAMBER_PRESSURE', 'MAIN_OUTER_AIR_BAG_PRESSURE',
       'CENTER_AIR_BAG_PRESSURE', 'RETAINER_RING_PRESSURE',
       'RIPPLE_AIR_BAG_PRESSURE', 'USAGE_OF_MEMBRANE',
       'USAGE_OF_PRESSURIZED_SHEET', 'SLURRY_FLOW_LINE_A',
       'SLURRY_FLOW_LINE_B', 'SLURRY_FLOW_LINE_C', 'WAFER_ROTATION',
       'STAGE_ROTATION', 'HEAD_ROTATION', 'DRESSING_WATER_STATUS',
       'EDGE_AIR_BAG_PRESSURE']



def grp_cpr1(s):
    row = { 'WAFER_ID': [s['WAFER_ID'].unique()[0]],
            'PATH_CHAMBERS': [len(s['CHAMBER'].unique())],
            'PATH_STAGES': [len(s['STAGE'].unique())],
            'START_TIME': [s['TIMESTAMP'].min()],
            'RUN_TIME': [s['TIMESTAMP'].max() - s['TIMESTAMP'].min()],
            'CHAMBER': [np.sum(list(s['CHAMBER'].unique()))],
        }

    for i in inte_features:
        row[i] = trapz(s[i], s['TIMESTAMP'])/row['RUN_TIME'][0]

    return pd.DataFrame.from_dict(row)



def grp_cpr2(s):
    s = s.sort_values(by='TIMESTAMP')
    s.drop_duplicates(subset=['TIMESTAMP'], inplace=True)
    cols = list(set(s.columns)-set(['TIMESTAMP']))
    
    X = s['TIMESTAMP']
    X_view = np.linspace(X.iloc[0], X.iloc[-1], 50)
    s2 = pd.DataFrame(X_view, columns=['TIMESTAMP'])
    for i in cols:
        tck = interpolate.make_interp_spline(x=X, y=s[i], k=1)
        piecewise_polynomial = interpolate.PPoly.from_spline(tck, extrapolate=None)
        s2[i] = piecewise_polynomial(X_view)

    s2['START_TIME'] = s['TIMESTAMP'].min()

    return s2


def sel_runpath(data, runpath):
    if runpath==0:
        chambers = [1,2,3,4,5,6]
        stages = ['A', 'B']
    elif runpath==1:
        chambers = [4,5,6]
        stages = ['A']
    elif runpath==2:
        chambers = [4,5,6]
        stages = ['B']
    elif runpath==3:
        chambers = [1,2,3]
        stages = ['A']

    return data[data['CHAMBER'].isin(chambers)&data['STAGE'].isin(stages)]


def data_preprocess(data, outp_name, runpath, kind=1):
    data = sel_runpath(data, runpath)
    data['STAGE'] = data['STAGE'].map(lambda x: 1 if x == 'A' else 0)
    
    if kind == 1:
        prep_grp = data.groupby(['WAFER_ID'], as_index=False).apply(grp_cpr1)
    elif kind == 2:
        prep_grp = data.groupby(['WAFER_ID'], as_index=False).apply(grp_cpr2)

    prep_grp = prep_grp.groupby(by='START_TIME', as_index=False, group_keys=True).apply(lambda s: s.sort_values(by='TIMESTAMP'))
    prep_grp.to_csv(outp_name, index=False)


train_set = pd.read_csv('data/train_data.csv', header=0)
test_set = pd.read_csv('data/test_data.csv', header=0)

data_preprocess(train_set, 'data/train_runpath1_cpr_50.csv', runpath=1, kind=2)
data_preprocess(test_set, 'data/test_runpath1_cpr_50.csv', runpath=1, kind=2)
