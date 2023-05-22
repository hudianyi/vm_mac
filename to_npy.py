import pandas as pd
import numpy as np

train_loc = 'data/train_runpath1_cpr_50.csv'
test_loc = 'data/test_runpath1_cpr_50.csv'

train_data = pd.read_csv(train_loc, header=0)
test_data = pd.read_csv(test_loc, header=0)


drop_features = ['TIMESTAMP', 'START_TIME', 'STAGE', 'WAFER_ID', 'MACHINE_DATA',  'CHAMBER', 'MACHINE_ID', 'AVG_REMOVAL_RATE', 'USAGE_OF_BACKING_FILM', 'USAGE_OF_PRESSURIZED_SHEET']
sele_features = list(set(test_data.columns)-set(drop_features))
print(sele_features)


def wafer_to_npy(s, a, b):
    s.sort_values(by='TIMESTAMP', inplace=True)
    b.append(s['AVG_REMOVAL_RATE'].mean())
    s = s[sele_features]
    a.append(s.to_numpy())


def to_npy(data, outp_name_x, outp_name_y):
    #for i in sele_features:
        #data[i] = (data[i]-data[i].min())/(data[i].max()-data[i].min())

    x_list = []
    y_list = []
    data.groupby('START_TIME').apply(wafer_to_npy, a=x_list, b=y_list)

    x_list = np.array(x_list)
    y_list = np.array(y_list)
    y_list = y_list.reshape(-1,1)

    np.save(outp_name_x, x_list)
    np.save(outp_name_y, y_list)


   
to_npy(train_data, 'data/trainx_runath1_50.npy', 'data/trainy_runath1_50.npy')
to_npy(test_data, 'data/testx_runath1_50.npy', 'data/testy_runath1_50.npy')
