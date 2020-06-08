from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

from keras.utils import to_categorical
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

def read_data(dataroot,file_ending='*.pcap_ISCX.csv'):
    if file_ending==None:
        print("please specify file ending pattern for glob")
        exit()
    print(join(dataroot,file_ending))
    filenames = [i for i in gb.glob(join(dataroot,file_ending))]
    combined_csv = pd.concat([pd.read_csv(f,dtype=object) for f in filenames],sort=False)
    return combined_csv

def cicids_data(params):
    dataroot="/home/vibek/Anomanly_detection_packages/CISIDS-2017/MachineLearningCSV/MachineLearningCVE_file/"
    data_path = read_data(dataroot,'*.pcap_ISCX.csv')
    num_records,num_features = data_path.shape
    print("there are {} flow records with {} feature dimension".format(num_records,num_features))

# there is white spaces in columns names e.g. ' Destination Port'
# So strip the whitespace from  column names
    data = data_path.rename(columns=lambda x: x.strip())
    print('stripped column names')

    df_label = data['Label']
    data = data.drop(columns=['Flow Packets/s','Flow Bytes/s','Label'])
    print('dropped bad columns')

    groups = xai.imbalance_plot(data, "Label", "Flow Duration")
    
    nan_count = data.isnull().sum().sum()
    print('There are {} nan entries'.format(nan_count))

    if nan_count>0:
      data.fillna(data.mean(), inplace=True)
      print('filled NAN')

# Normalising all numerical features:
    cols_to_norm = list(data.columns.values)[:76]  
    #print('cols_to_norm:\n', cols_to_norm)
    data = data.astype(np.float32)
    print(data)
    
    mask = data ==-1
    data[mask]=0
    
#  to leave -1 (missing features) values as is and exclude in normilizing
    mean_i = np.mean(data,axis=0)
    min_i = np.min(data,axis=0) 
    max_i = np.max(data,axis=0)
# zero centered 
    r = max_i-min_i+eps
    data = (data-mean_i)/r  

#deal with missing features -1
    data[mask] = 0 

    data = data.astype(float).apply(pd.to_numeric)
    print('converted to numeric')

# lets count if there is NaN values in our dataframe(missing features)
    assert data.isnull().sum().sum()==0, "There should not be any NaN values"
    X = data.values
    
# To encode string  labels into numbers
    df_label = df_label.apply(lambda x : transform(x))
    le = LabelEncoder()
    Y = le.fit_transform(df_label)
    Y = to_categorical(Y)
    classe_names = list(le.classes_)
# Create training and test sets
    train_data, train_labels, test_data, test_labels = train_test_split(X,Y,test_size=0.3, random_state=seed)
    return train_data, train_labels, test_data, test_labels

def cicids_data_common(params):
    dataroot="/home/vibek/Anomanly_detection_packages/CISIDS-2017/MachineLearningCSV/MachineLearningCVE/"
    data_path = read_data(dataroot,'*.pcap_ISCX.csv')
    num_records,num_features = data_path.shape
    print("there are {} flow records with {} feature dimension".format(num_records,num_features))

    # there is white spaces in columns names e.g. ' Destination Port'
# So strip the whitespace from  column names
    data = data_path.rename(columns=lambda x: x.strip())
    print('stripped cicids column names')

    df_label = data['Label']
    #print('df_label:\n', df_label)

    data = data.drop(columns=['Flow ID','Fwd Header Length.1','Protocol','Timestamp','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets','Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean','Fwd Packet Length Std','Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std','Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Total','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags','Fwd Header Length','Bwd Header Length','Min Packet Length','Max Packet Length','Packet Length Mean','Packet Length Std','Packet Length Variance','FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count','Down/Up Ratio','Average Packet Size','Avg Fwd Segment Size','Avg Bwd Segment Size','Fwd Header Length','Fwd Avg Bytes/Bulk','Fwd Avg Packets/Bulk','Fwd Avg Bulk Rate','Bwd Avg Bytes/Bulk','Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Init_Win_bytes_forward','Init_Win_bytes_backward','act_data_pkt_fwd','min_seg_size_forward','Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min','Label'])
    print('dropped bad columns')

    nan_count = data.isnull().sum().sum()
    print('There are {} nan entries'.format(nan_count))

    if nan_count>0:
      data.fillna(data.mean(), inplace=True)
      print('filled NAN')

    #lb = LabelBinarizer()
    le = LabelEncoder()

    data['Source IP'] = le.fit_transform(data['Source IP'])
    data['Destination IP'] = le.fit_transform(data['Destination IP'])

    # Normalising all numerical features:
    cols_to_norm = list(data.columns.values)[:11]  
    #print('cols_to_norm:\n', cols_to_norm)
    data = data.astype(np.float32)
    #print('data type:\n', data.shape)
    mask = data ==-1
    data[mask]=0

    #  to leave -1 (missing features) values as is and exclude in normilizing
    mean_i = np.mean(data,axis=0)
    min_i = np.min(data,axis=0) 
    max_i = np.max(data,axis=0)
# zero centered 
    r = max_i-min_i+eps
    data = (data-mean_i)/r  

#deal with missing features -1
    data[mask] = 0 

    data = data.astype(float).apply(pd.to_numeric)
    print('converted to numeric')

    #histogram to see the data distribution
    data.hist(figsize=(15,18))
    plt.show()

# lets count if there is NaN values in our dataframe(missing features)
    assert data.isnull().sum().sum()==0, "There should not be any NaN values"
    X = data.values

    # To encode string  labels into numbers
    df_label = df_label.apply(lambda x : transform(x))
    #le = LabelEncoder()
    Y = le.fit_transform(df_label)
    Y = to_categorical(Y)
    classe_names = list(le.classes_)
# Create training and test sets
    train_data, train_labels, test_data, test_labels = train_test_split(X,Y,test_size=0.35, random_state=seed)
    return train_data, train_labels, test_data, test_labels


#encoding label in binary 0: BENIGN and 1:Attack
def transform(x):
    if x == "BENIGN":
        return 0
    else:
        return 1

#We balance data as follows:
#1) oversample small classes so that their population/count is equal to mean_number_of_samples_per_class
#2) undersample large classes so that their count is equal to mean_number_of_samples_per_class
def balance_data(X,Y,seed):
    np.random.seed(seed)
    unique,counts = np.unique(y,return_counts=True)
    mean_samples_per_class = int(round(np.mean(counts)))
#(number of examples, number of features)    
    N,D = X.shape 
    new_X = np.empty((0,D)) 
    new_y = np.empty((0),dtype=int)
    for i,c in enumerate(unique):
        temp_x = X[y==c]
# gets `mean_samples_per_class` indices of class `c`
        indices = np.random.choice(temp_x.shape[0],mean_samples_per_class) 
# now we put new data into new_X 
        new_X = np.concatenate((new_X,temp_x[indices]),axis=0) 
        temp_y = np.ones(mean_samples_per_class,dtype=int)*c
        new_y = np.concatenate((new_y,temp_y),axis=0)
        
# in order to break class order in data we need shuffling
    indices = np.arange(new_y.shape[0])
    np.random.shuffle(indices)
    new_X =  new_X[indices,:]
    new_y = new_y[indices]
    return (new_X,new_y)