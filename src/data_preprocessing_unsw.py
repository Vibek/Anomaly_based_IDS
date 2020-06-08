from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
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
# ***** UNSW STRING FEATURES VALUES *****
protocol_values = ['tcp', 'udp', 'arp', 'ospf', 'icmp', 'igmp', 'rtp', 'ddp',
                'ipv6-frag', 'cftp', 'wsn', 'pvp', 'wb-expak', 'mtp',
                'pri-enc', 'sat-mon', 'cphb', 'sun-nd', 'iso-ip', 'xtp', 'il',
                'unas', 'mfe-nsp', '3pc', 'ipv6-route', 'idrp', 'bna', 'swipe',
                'kryptolan', 'cpnx', 'rsvp', 'wb-mon', 'vmtp', 'ib', 'dgp',
                'eigrp', 'ax.25', 'gmtp', 'pnni', 'sep', 'pgm', 'idpr-cmtp',
                'zero', 'rvd', 'mobile', 'narp', 'fc', 'pipe', 'ipcomp',
                'ipv6-no', 'sat-expak', 'ipv6-opts', 'snp', 'ipcv',
                'br-sat-mon', 'ttp', 'tcf', 'nsfnet-igp', 'sprite-rpc',
                'aes-sp3-d', 'sccopmce', 'sctp', 'qnx', 'scps', 'etherip',
                'aris', 'pim', 'compaq-peer', 'vrrp', 'iatp', 'stp',
                'l2tp', 'srp', 'sm', 'isis', 'smp', 'fire', 'ptp', 'crtp',
                'sps', 'merit-inp', 'idpr', 'skip', 'any', 'larp', 'ipip',
                'micp', 'encap', 'ifmp', 'tp++', 'a/n', 'ipv6', 'i-nlsp',
                'ipx-n-ip', 'sdrp', 'tlsp', 'gre', 'mhrp', 'ddx', 'ippc',
                'visa', 'secure-vmtp', 'uti', 'vines', 'crudp', 'iplt',
                'ggp', 'ip', 'ipnip', 'st2', 'argus', 'bbn-rcc', 'egp',
                'emcon', 'igp', 'nvp', 'pup', 'xnet', 'chaos', 'mux', 'dcn',
                'hmp', 'prm', 'trunk-1', 'xns-idp', 'leaf-1', 'leaf-2', 'rdp',
                'irtp', 'iso-tp4', 'netblt', 'trunk-2', 'cbt']

flags_values = ['FIN', 'INT', 'CON', 'ECO', 'REQ', 'RST', 'PAR', 'URN', 'no',
                'ACC', 'CLO']

service_values = ['-', 'ftp', 'smtp', 'snmp', 'http', 'ftp-data',
                  'dns', 'ssh', 'radius', 'pop3', 'dhcp', 'ssl', 'irc']

# Dictionary that contains mapping of various attacks to the four main categories + normal

attack_cat_values = ['Normal', 'Backdoor', 'Analysis', 'Fuzzers', 'Shellcode',
                     'Reconnaissance', 'Exploits', 'DoS', 'Worms', 'Generic']

def unsw_data(params):
    # ***** DATA PATHS *****

    data_path = "/home/vibek/Anomanly_detection_packages/UNSW-NB/"
    train_data_path = data_path+"UNSW_NB15_training-set.csv"
    test_data_path = data_path+"UNSW_NB15_testing-set.csv"

    # Load csv data into dataframes without 'id' and 'Label'
    train_df = pd.read_csv(train_data_path).drop(['id'], axis=1)
    test_df = pd.read_csv(test_data_path).drop(['id'], axis=1)

    # Creates new dummy columns from each unique string in a particular feature
    training = pd.get_dummies(data=train_df, columns=['proto', 'service', 'state'])
    testing = pd.get_dummies(data=test_df, columns=['proto', 'service', 'state'])

    # Making sure that the training features are same as testing features. 
    traincols = list(training.columns.values)
    testcols = list(testing.columns.values)

    # For those in training but not in testing
    for col in traincols:
        # If a column is missing in the testing dataset, we add it
        if col not in testcols:
            testing[col] = 0
            testcols.append(col)
    # For those in testing but not in training
    for col in testcols:
        if col not in traincols:
            training[col] = 0
            traincols.append(col)


    # Moving the labels and categories to the end and making sure features are in the same order
    traincols.pop(traincols.index('attack_cat'))
    traincols.pop(traincols.index('label'))
    training = training[traincols + ['attack_cat', 'label']]
    testing = testing[traincols + ['attack_cat', 'label']]

   # To encode string  labels into numbers
    le = LabelEncoder()

    # Encoding the category names into numbers so that they can be one hot encoded later.
    training['attack_cat'] = le.fit_transform(training['attack_cat'])
    testing['attack_cat'] = le.fit_transform(testing['attack_cat'])

    # Normalising all numerical features:
    cols_to_norm = list(training.columns.values)[:39]
    training[cols_to_norm] = training[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    testing[cols_to_norm] = testing[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    #print('cols_to_norm:\n',cols_to_norm)

    classe_names = list(le.classes_)
    #print(classe_names)
    training = training.to_numpy(dtype='float32')[:-237, :]
    train_data = training[:, :-2]
    train_labels = training[:, -1].astype(np.float32)
    #train_classes = training[:, -2].astype(np.float32)
    testing = testing.to_numpy(dtype='float32')
    test_data = testing[:, :-2]
    test_labels = testing[:, -1].astype(np.float32)
    #test_classes = testing[:, -2].astype(np.float32)

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_data, train_labels, test_data, test_labels

    #data preprocessing UNSW-NB15 data
def read_data(dataroot,file_ending='*_unsw.csv'):
    if file_ending==None:
        print("please specify file ending pattern for glob")
        exit()
    print(join(dataroot,file_ending))
    filenames = [i for i in gb.glob(join(dataroot,file_ending))]
    combined_csv = pd.concat([pd.read_csv(f,dtype=object) for f in filenames],sort=False)
    return combined_csv

def unsw_data_common(params):
    dataroot="/home/vibek/Anomanly_detection_packages/UNSW-NB/"
    data_path = read_data(dataroot,'*_unsw.csv')
    num_records,num_features = data_path.shape
    print("there are {} flow records with {} feature dimension".format(num_records,num_features))

    # there is white spaces in columns names e.g. ' Destination Port'
# So strip the whitespace from  column names
    data = data_path.rename(columns=lambda x: x.strip())
    print('stripped unsw column names')
    #_ = xai.correlations(data, include_categorical=True)

    df_label = data['Label']

    #groups = xai.imbalance_plot(data, "attack_cat", "dur")

    data = data.drop(columns=['proto','state','sloss','dloss','service','Sload','Dload','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','Label'])
    print('dropped bad columns')

    nan_count = data.isnull().sum().sum()
    print('There are {} nan entries'.format(nan_count))

    if nan_count>0:
      data.fillna(data.mean(), inplace=True)
      print('filled NAN')

    le = LabelEncoder()

    data['srcip'] = le.fit_transform(data['srcip'])
    data['dstip'] = le.fit_transform(data['dstip'])
    data['dur'] = le.fit_transform(data['dur'])
    data['sport'] = le.fit_transform(data['sport'])
    data['dsport'] = le.fit_transform(data['dsport'])
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
    data.hist(figsize=(15,18))
    plt.show()

# lets count if there is NaN values in our dataframe(missing features)
    assert data.isnull().sum().sum()==0, "There should not be any NaN values"
    X = data.values

    # To encode string  labels into numbers
    #df_label_final = df_label_final.apply(lambda x : transform(x))
    #le = LabelEncoder()
    #Y = le.fit_transform(df_label_final)
    Y = to_categorical(df_label)
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
