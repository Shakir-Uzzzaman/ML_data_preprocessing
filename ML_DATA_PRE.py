import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import ast
 

datafile = pd.read_csv(r'F:\machine learning\pre\2OBJECT.csv')

datafile = datafile.dropna(subset=['CSI_DATA'])





# Converting string to list
#Y = ast.literal_eval(Y)


mean_val=datafile['rate'].mean()
datafile['rate'].fillna(value=mean_val,inplace=True)

mean_val=datafile['sig_mode'].mean()
datafile['sig_mode'].fillna(value=mean_val,inplace=True)

mean_val=datafile['mcs'].mean()
datafile['mcs'].fillna(value=mean_val,inplace=True)

mean_val=datafile['bandwidth'].mean()
datafile['bandwidth'].fillna(value=mean_val,inplace=True)

mean_val=datafile['smoothing'].mean()
datafile['smoothing'].fillna(value=mean_val,inplace=True)

mean_val=datafile['not_sounding'].mean()
datafile['not_sounding'].fillna(value=mean_val,inplace=True)


mean_val=datafile['aggregation'].mean()
datafile['aggregation'].fillna(value=mean_val,inplace=True)

mean_val=datafile['stbc'].mean()
datafile['stbc'].fillna(value=mean_val,inplace=True)

mean_val=datafile['fec_coding'].mean()
datafile['fec_coding'].fillna(value=mean_val,inplace=True)


mean_val=datafile['sgi'].mean()
datafile['sgi'].fillna(value=mean_val,inplace=True)


mean_val=datafile['noise_floor'].mean()
datafile['noise_floor'].fillna(value=mean_val,inplace=True)


mean_val=datafile['ampdu_cnt'].mean()
datafile['ampdu_cnt'].fillna(value=mean_val,inplace=True)


mean_val=datafile['channel'].mean()
datafile['channel'].fillna(value=mean_val,inplace=True)


mean_val=datafile['secondary_channel'].mean()
datafile['secondary_channel'].fillna(value=mean_val,inplace=True)

mean_val=datafile['sig_mode'].mean()
datafile['sig_mode'].fillna(value=mean_val,inplace=True)


mean_val=datafile['local_timestamp'].mean()
datafile['local_timestamp'].fillna(value=mean_val,inplace=True)

mean_val=datafile['ant'].mean()
datafile['ant'].fillna(value=mean_val,inplace=True)


mean_val=datafile['sig_len'].mean()
datafile['sig_len'].fillna(value=mean_val,inplace=True)

mean_val=datafile['sig_mode'].mean()
datafile['sig_mode'].fillna(value=mean_val,inplace=True)


mean_val=datafile['rx_state'].mean()
datafile['rx_state'].fillna(value=mean_val,inplace=True)


mean_val=datafile['real_time_set'].mean()
datafile['real_time_set'].fillna(value=mean_val,inplace=True)


mean_val=datafile['real_timestamp'].mean()
datafile['real_timestamp'].fillna(value=mean_val,inplace=True)

n=datafile.isnull().sum()


mean_val=datafile['len'].mean()
datafile['len'].fillna(value=mean_val,inplace=True)

X = datafile.iloc[:,:-1].values
Y = datafile.iloc[:,25].values






from sklearn.preprocessing import LabelEncoder
lblencode = LabelEncoder()
X[:,0] = lblencode.fit_transform(X[:,0])
X[:,1] = lblencode.fit_transform(X[:,1])
X[:,2] = lblencode.fit_transform(X[:,2])





print(n)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=10)

Train_dataframe= pd.DataFrame(X_test)
Train_dataframe['CSI_DATA'] = Y_test

Train_dataframe.to_csv('test.csv')

 
# save the dataframe as a csv file

'''
from sklearn.preprocessing import StandardScaler
stdscalar = StandardScaler()
X_train = stdscalar.fit_transform(X_train)
X_test = stdscalar.transform(X_test)
'''


























