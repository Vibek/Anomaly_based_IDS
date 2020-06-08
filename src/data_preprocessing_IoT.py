#sklearn libraries
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

def read_data(dataroot,file_ending='*.labeled.csv'):
    if file_ending==None:
        print("please specify file ending pattern for glob")
        exit()
    print(join(dataroot,file_ending))
    filenames = [i for i in gb.glob(join(dataroot,file_ending))]
    combined_csv = pd.concat([pd.read_csv(f,dtype=object) for f in filenames],sort=False)
    return combined_csv

def IoT_data(params):
    dataroot="/home/vibek/Anomanly_detection_packages/IoT-23/IoTScenarios/"
    data_path = read_data(dataroot,'*.labeled.csv')
    num_records,num_features = data_path.shape
    print("there are {} flow records with {} feature dimension".format(num_records,num_features))

# there is white spaces in columns names e.g. ' Destination Port'
# So strip the whitespace from  column names
    data = data_path.rename(columns=lambda x: x.strip())
    print('stripped column names')

    df_label = data['label_s']
    print('df_label:\n', df_label)
    
    df_label_1 = data['label_s'].str.replace('0   benign   0', 'BENIGN')
    df_label_2 = df_label_1.str.replace('(empty)   Benign   0', 'BENIGN')
    df_label_final = df_label_2.str.replace('(empty)   Malicious   PartOfAHorizontalPortScan', 'Malicious')
    print('df_label new:\n', df_label_final)

    data = data.drop(columns=['id','proto_s','service_s','sip_s','dip_s','history_s','connstate_s','label_s'])
    print('dropped bad columns')
    
    nan_count = data.isnull().sum().sum()
    print('There are {} nan entries'.format(nan_count))

    if nan_count>0:
      data.fillna(data.mean(), inplace=True)
      print('filled NAN')

# Normalising all numerical features:
    cols_to_norm = list(data.columns.values)[:14]  
    print('cols_to_norm:\n', cols_to_norm)
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

# lets count if there is NaN values in our dataframe(missing features)
    assert data.isnull().sum().sum()==0, "There should not be any NaN values"
    X = data.values
    print("Value of X:\n", X.shape)
    
# To encode string  labels into numbers
    df_label_final = df_label_final.apply(lambda x : transform(x))
    print("Value of df_label:\n", df_label_final.shape)
    le = LabelEncoder()
    Y = le.fit_transform(df_label_final)
    Y = to_categorical(Y)
    print('Value of Y:', Y.shape)
    classe_names = list(le.classes_)
    print('classe_names:\n', classe_names)
# Create training and test sets
    train_data, train_labels, test_data, test_labels = train_test_split(X,Y,test_size=0.3, random_state=seed)
    return train_data, train_labels, test_data, test_labels

def IoT_data_common(params):
    dataroot="/home/vibek/Anomanly_detection_packages/IoT-23/IoTScenarios/"
    data_path = read_data(dataroot,'*.labeled.csv')
    num_records,num_features = data_path.shape
    print("there are {} flow records with {} feature dimension".format(num_records,num_features))

    # there is white spaces in columns names e.g. ' Destination Port'
# So strip the whitespace from  column names
    data = data_path.rename(columns=lambda x: x.strip())
    print('stripped IoT column names')

    categorical_cols = ["sip_s", "sport_s", "dip_s", "duration_f", "dport_s",
                   "ibytes_f", "obyte_f", "ipkt_f", "iipbytes_f", "opkt_f", "opkt_f","label_s"]

    df_label = data['label_s']
    ax=plt.figure(figsize=(12,3))
    sns.countplot(data['label_s'])
    plt.show()
    #print('df_label:\n', df_label)
    #df_groups = xai.imbalance_plot(data, "label_s", categorical_cols=categorical_cols)
    #groups = xai.imbalance_plot(data, "label_s", "dport_s", categorical_cols=categorical_cols)
    #bal_df = xai.balance(data, "label_s", "dport_s", upsample=0.8, categorical_cols=categorical_cols)
    #_ = xai.correlations(data, include_categorical=True)

    df_label_1 = data['label_s'].str.replace('0   benign   0', 'BENIGN')
    df_label_2 = df_label_1.str.replace('(empty)   Benign   0', 'BENIGN')
    df_label_final = df_label_2.str.replace('(empty)   Malicious   PartOfAHorizontalPortScan', 'Malicious')
    #print('df_label new:\n', df_label_final)

    data = data.drop(columns=['ts_f','id','proto_s','service_s','connstate_s','ilocal_f','olocal_f','missedbytes_f','history_s','label_s'])
    print('dropped bad columns')

    #data_common = data['sip_s','sport_s','dip_s','dport_s','duration_f','ibytes_f','obyte_f','ipkt_f','iipbytes_f','opkt_f','oipbytes_f']
    #print('select common columns:\n', data_common)

    nan_count = data.isnull().sum().sum()
    print('There are {} nan entries'.format(nan_count))

    if nan_count>0:
      data.fillna(data.mean(), inplace=True)
      print('filled NAN')

    #lb = LabelBinarizer()
    le = LabelEncoder()

    data['sip_s'] = le.fit_transform(data['sip_s'])
    #print(data)

    data['dip_s'] = le.fit_transform(data['dip_s'])
    #print(data)

    # Normalising all numerical features:
    cols_to_norm = list(data.columns.values)[:11]  
    #print('cols_to_norm:\n', cols_to_norm)
    data = data.astype(np.float32)
    #print('data type:\n', data)
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
    data.hist(figsize=(3,5))
    plt.show()

# lets count if there is NaN values in our dataframe(missing features)
    assert data.isnull().sum().sum()==0, "There should not be any NaN values"
    X = data.values

    # To encode string  labels into numbers
    df_label_final = df_label_final.apply(lambda x : transform(x))
    #le = LabelEncoder()
    Y = le.fit_transform(df_label_final)
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
    unique,counts = np.unique(Y,return_counts=True)
    mean_samples_per_class = int(round(np.mean(counts)))
#(number of examples, number of features)    
    N,D = X.shape 
    new_X = np.empty((0,D)) 
    new_y = np.empty((0),dtype=int)
    for i,c in enumerate(unique):
        temp_x = X*c
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
