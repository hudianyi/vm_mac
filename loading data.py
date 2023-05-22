# -*- coding: utf-8 -*-

import pandas as pd
import os


PATH = '2016 PHM DATA CHALLENGE CMP DATA SET/CMP-data'
train_xpath = os.path.join(PATH, "training")
test_xpath = os.path.join(PATH, "test")

train_ypath = "2016 PHM DATA CHALLENGE CMP DATA SET/CMP-training-removalrate.csv"
test_ypath = "2016 PHM DATA CHALLENGE CMP DATA SET/CMP-test-removalrate.csv"


def load_xdata(path):
    df_list = []
    for file_name in os.listdir(path):
        df = pd.read_csv(os.path.join(path, file_name))
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)
    

train_xdata = load_xdata(train_xpath)
train_ydata = pd.read_csv(train_ypath, header=0)
train_data = pd.merge(train_xdata, train_ydata)


test_xdata = load_xdata(test_xpath)
test_ydata = pd.read_csv(test_ypath, header=0)
test_data = pd.merge(test_xdata, test_ydata)


train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)