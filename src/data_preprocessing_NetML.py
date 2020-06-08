#python libraries
import os
import json
import gzip
import time as t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import glob as gb

#XAI libraries
import xai
import xai.data
import seaborn as sns

#sklearn libraries
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels

def read_json_gz(jsonFilename, featureDict=None):
    feature_header = []
    # Open json file from gzip
    with gzip.open(jsonFilename, "rb") as jj:
        #data = [json.loads(line) for line in jj]
        # Write a for loop and check every single flow with utf-8:
        data = []
        enc = []
        pb_dataline = []

        i = 0
        while True:
            i += 1
            try:
                flow = jj.readline().decode("utf-8") # decoded to convert bytes to str for JSON.
                if not flow:
                    break
                sample = json.loads(flow)
                data.append(sample)
            except:
            	pb_dataline.append(i)
            	print("Line {} has invalid character. Skipped ...".format(i))
        if len(pb_dataline) != 0:
            print("Total {} lines were skipped because of invalid characters.".format(len(pb_dataline)))

        if featureDict is None:
            with open("/home/vibek/Anomanly_detection_packages/NetML-2020/featureDict_META.json", 'r') as js:
                featureDict = json.load(js)

        # Create an empty numpy array of arbitrarily but sufficiently large (2048) in terms of columns
        dataArray = np.zeros((len(data), 2048))

        # Compare len(feature) for each flow, if greater in current flow, then add it to feature_header
        max_len_features = 0
        # Retrieve the selected features and fill the dataArray.
        ids = []
        for i in range(len(data)):
            ids.append(data[i]['id']) 
            # Count the number of columns to truncate at last
            colCounter = 0
            for feature in sorted(featureDict.keys()):
                extracted = data[i][feature]
                if type(extracted) is list:
                    if len(extracted) > 0:
                        # SPLT and byte_dist is in dict format. skip for now
                        if type(extracted[0]) == dict:
                            #print("SPLT and byte_dist is in dict format. skip for now.")
                            pass # To supress print
                        # If all selected (i.e. == -1) then return all
                        elif featureDict[feature] == -1: 
                            for j in range(len(extracted)):
                                dataArray[i, colCounter] = extracted[j]
                                if len(list(data[i].keys())) > max_len_features: 
                                    if feature+"_"+str(j) not in feature_header:
                                        feature_header.append(feature+"_"+str(j))
                                # Update colCounter by 1
                                colCounter += 1
                        # If only some indices are selected
                        else: 
                            for j in featureDict[feature]:
                                dataArray[i, colCounter] = extracted[j]
                                if len(list(data[i].keys())) > max_len_features:
                                    if feature+"_"+str(j) not in feature_header:
                                        feature_header.append(feature+"_"+str(j))
                                # Update colCounter by 1
                                colCounter += 1
                # If extracted feature is not list but a single value
                elif type(extracted) is str:
                    #print(feature + ": " + extracted + " is skipped because it has str type data.")
                    pass # To supress print
                else:
                    dataArray[i, colCounter] = extracted
                    if len(list(data[i].keys())) > max_len_features:
                        if feature not in feature_header:
                            feature_header.append(feature)
                    # Increase colCounter by 1
                    colCounter += 1
            
            # Update max_len_features for next flow
            if len(list(data[i].keys())) > max_len_features:
                max_len_features = len(list(data[i].keys()))
        # Truncate dataArray to the actual columnsize = colCounter and return
        return dataArray[:,:colCounter], ids, feature_header


def read_NetML_dataset(datasetFolderName, annotationFileName=None, class_label_pairs=None):
    # Training          : data, anno, clp=None (returns clp)
    # Test-with anno    : data, anno, clp=Returned_from_training
    # Prediction:       : data, anno=None, clp=Returned_from_training

    label = []
    # dataArray initialize
    dataArray = None
    # feature_names initialize
    feature_names = []

    for root, dirs, files in os.walk(datasetFolderName): 
        for f in files:                        
            if f.endswith((".json.gz")):
                print("Reading {}".format(f))
                #try:
                d, ids, f_names = read_json_gz(os.path.join(root, f))
                #except:
                #    print("File {} is errorenous! Skipped.".format(f))
                
                # Check if f_names has more features
                if len(f_names) > len(feature_names):
                    feature_names = f_names
                if dataArray is None:
                    dataArray = d
                else:
                    dataArray = np.concatenate((dataArray, d), axis=0)

                if annotationFileName is not None:
                    with gzip.open(annotationFileName, "rb") as an:
                        anno = json.loads(an.read().decode("utf-8")) 

                    for i in range(d.shape[0]):
                        id_str = str(ids[i])
                        label.append(anno[id_str])

    # Training or test-with anno case, return labels
    if annotationFileName is not None: 
        labelArray, class_label_pairs = encode_label(label)
        ax=plt.figure(figsize=(12,3))
        sns.countplot(label)
        plt.show()
        #print("shape of labelArray: ", label)
        
        return feature_names, ids, dataArray, labelArray, class_label_pairs
    
    # Prediction case, no return of labelArray and class_label_pairs
    else: 
        #print("shape of dataArray: ", dataArray.shape)
        
        return feature_names, ids, dataArray, 0, 0

def read_anno_json_gz(filename, class_label_pairs=None):
    # Read annotation JSON.gz file:
    with gzip.open(filename, "rb") as an:
        anno = json.loads(an.read().decode('utf-8'))

    # Sort ids by ascending order
    anno_sorted = {}
    for k in sorted(anno.keys()):
        anno_sorted[k] = anno[k]

    # Encode the labels to integer values from 0 to n_classes-1
    y_, class_label_pairs = encode_label(list(anno_sorted.values()), class_label_pairs)
    
    return y_, class_label_pairs   

def get_training_data(training_set_foldername, anno_file_name):
    # Read training set from json files
    print("\nLoading training set ...")
    training_feature_names, ids, training_data, training_label, training_class_label_pair = read_NetML_dataset(training_set_foldername, anno_file_name, class_label_pairs=None)
    num_records,num_features = training_data.shape
    print("there are {} flow records with {} feature dimension".format(num_records,num_features))
    #print('Training feature_names:\n', training_feature_names)  
  
    # Convert np.array to dataframe for easy manipulations
    training_df = pd.DataFrame(data=training_data,                          # values
                        index=[i for i in range(training_data.shape[0])],   # 1st column as index
                        columns=training_feature_names)                     # 1st row as the column names

    #histogram to see the data distribution
    training_df.hist(figsize=(300,300))
    plt.show()
    # Get values for Xtrain
    Xtrain = training_df.values
    #print('Value of data:\n', Xtrain)
    
    return Xtrain, training_label, training_class_label_pair, ids


def get_testing_data(test_set_foldername):
    # Read test set from json files
    print("Loading testing set ...")
    test_feature_names, ids, test_data, _, _, = read_NetML_dataset(test_set_foldername)

    # Convert np.array to dataframe for easy manipulations
    test_df = pd.DataFrame(data=test_data,                              # values
                        index=[i for i in range(test_data.shape[0])],   # 1st column as index
                        columns=test_feature_names)                     # 1st row as the column names


    # Get np.array for Xtest
    Xtest = test_df.values

    return Xtest, ids


def make_result_submission(user_annotations, ids, class_label_pairs, filepath):
    output = {}
    for i in range(user_annotations.shape[0]):
        output[str(ids[i])] = [k for k, v in class_label_pairs.items() if v == user_annotations[i]][0]

    with open(filepath, "w") as jf:
        json.dump(output, jf, indent=4)
    print("Testing file is created as .{}\n".format(filepath[filepath.find("/home/vibek/Anomanly_detection_packages/DNN_Package/results"):]))


def encode_label(labels, class_label_pairs=None):

    unique_labels = []
    label_list = []
    clp = []
    if class_label_pairs is None:
        class_label_pairs = {}
        [unique_labels.append(label) for label in labels if label not in unique_labels]
        unique_labels.sort()
        l = 0
        for ul in unique_labels:
            class_label_pairs[ul] = l
            l += 1

    [label_list.append(class_label_pairs[label]) for label in labels]

    #for label in unique_labels:
    #print('unique_labels:\n',unique_labels)
    labelArray = np.asarray(label_list).reshape((-1,))

    return labelArray, class_label_pairs


def one_hot(y_, n_classes=None):
    # Function to encode neural one-hot output labels from number indexes
    if n_classes is None:
        n_classes = int(int(max(y_))+1)
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS
