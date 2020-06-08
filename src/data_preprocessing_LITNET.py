from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

#keras libraries
from keras.utils import to_categorical

#python libraries
from os.path import join
import numpy as np
import pandas as pd
import glob as gb
import xai
import xai.data
import seaborn as sns
import matplotlib.pyplot as plt

#global variable
seed = 124
eps = 1e-15

def LITNET_data(params):
    data_path = "/home/vibek/Anomanly_detection_packages/LITNET-2020/ALLinONE/"
    data_path_csv = data_path+"allFlows.csv"

     # Load csv data into dataframes without 'id' and 'Label'
    data = pd.read_csv(data_path_csv, nrows=1099999)# I took first 1099999 rows, change according to your need
    num_records,num_features = data.shape
    print("there are {} flow records with {} feature dimension".format(num_records,num_features))

# there is white spaces in columns names e.g. ' Destination Port'
# So strip the whitespace from  column names
    data = data.rename(columns=lambda x: x.strip())
    print('stripped column names')

    df_label = data['attack_a']
    print('df_label:\n', df_label)

    ax=plt.figure(figsize=(12,3))
    sns.countplot(data['attack_t'])
    plt.show()
 
    data = data.drop(columns=['ID','ts_year', 'ts_month', 'ts_day', 'ts_hour', 'ts_min', 'ts_second', 'te_year', 'te_month', 'te_day', 'te_hour', 'te_min', 'te_second', 'pr','_flag1','_flag2','_flag3','_flag4','_flag5','_flag6', 'nh', 'nhb', 'ismc', 'odmc', 'idmc', 'osmc', 'mpls1', 'mpls2','mpls3','mpls4','mpls5','mpls6','mpls7','mpls8','mpls9','mpls10','ra', 'eng', 'tr', 'tcp_f_n_a', 'tcp_f_n_f', 'tcp_f_n_r', 'tcp_f_n_p', 'tcp_f_n_u', 'tcp_src_tftp', 'tcp_src_kerb', 'tcp_src_rpc', 'attack_t', 'attack_a'])
    print('dropped bad columns')
    print('Data:\n', data)
    #return data
    
    nan_count = data.isnull().sum().sum()
    print('There are {} nan entries'.format(nan_count))

    if nan_count>0:
      data.fillna(data.mean(), inplace=True)
      print('filled NAN')

# Normalising all numerical features:
    le = LabelEncoder()

    data['sa'] = le.fit_transform(data['sa'])
    data['da'] = le.fit_transform(data['da'])
    data['icmp_dst_ip_b'] = le.fit_transform(data['icmp_dst_ip_b'])
    data['icmp_src_ip'] = le.fit_transform(data['icmp_src_ip'])
    data['udp_dst_p'] = le.fit_transform(data['udp_dst_p'])
    data['tcp_f_s'] = le.fit_transform(data['tcp_f_s'])
    #data['tcp_f_n_a'] = le.fit_transform(data['tcp_f_n_a'])
    #data['tcp_f_n_f'] = le.fit_transform(data['tcp_f_n_f'])
    #data['tcp_f_n_r'] = le.fit_transform(data['tcp_f_n_r'])
    #data['tcp_f_n_p'] = le.fit_transform(data['tcp_f_n_p'])
    #data['tcp_f_n_u'] = le.fit_transform(data['tcp_f_n_u'])
    data['tcp_dst_p'] = le.fit_transform(data['tcp_dst_p'])
    data['tcp_src_dst_f_s'] = le.fit_transform(data['tcp_src_dst_f_s'])
    #data['tcp_src_tftp'] = le.fit_transform(data['tcp_src_tftp'])
    #data['tcp_src_kerb'] = le.fit_transform(data['tcp_src_kerb'])
    #data['tcp_src_rpc'] = le.fit_transform(data['tcp_src_rpc'])
    data['tcp_dst_p_src'] = le.fit_transform(data['tcp_dst_p_src'])
    data['smtp_dst'] = le.fit_transform(data['smtp_dst'])
    data['udp_p_r_range'] = le.fit_transform(data['udp_p_r_range'])
    data['p_range_dst'] = le.fit_transform(data['p_range_dst'])
    data['udp_src_p_0'] = le.fit_transform(data['udp_src_p_0'])

    print('Data_new:\n', data)

    data = data.astype(np.float32)
    print('data type:\n', data)
    mask = data ==-1
    data[mask]=0
    
#  to leave -1 (missing features) values as is and exclude in normilizing
    mean_i = np.mean(data,axis=0)
    min_i = np.min(data,axis=0) 
    max_i = np.max(data,axis=0)
# zero centered 
    r = (max_i-min_i)+eps
    data = (data-mean_i)/r  

#deal with missing features -1
    data[mask] = 0 

    data = data.astype(float).apply(pd.to_numeric)
    print('converted to numeric\n', data)

    #histogram to see the data distribution
    data.hist(figsize=(300,300))
    plt.show()

# lets count if there is NaN values in our dataframe(missing features)
    assert data.isnull().sum().sum()==0, "There should not be any NaN values"
    X = data.values
    print("Value of X:\n", X.shape)
    
# To encode string  labels into numbers
    Y = to_categorical(df_label)
    print('Value of Y:', Y.shape)
    classe_names = list(le.classes_)
    print('classe_names:\n', classe_names)
# Create training and test sets
    train_data, train_labels, test_data, test_labels = train_test_split(X,Y,test_size=0.35, random_state=seed)
    return train_data, train_labels, test_data, test_labels
