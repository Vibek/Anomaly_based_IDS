from __future__ import print_function

#Python libraries
from datetime import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot
import io
import itertools
import pickle
import shutil
import time
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output

##Sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             precision_score, auc, recall_score, f1_score)
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.metrics import make_scorer, auc
from sklearn.metrics import classification_report
import scikitplot as skplt
import h5py

#keras libraries
from keras.models import load_model
from keras.models import model_from_json
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from livelossplot.keras import PlotLossesCallback
from keras.models import Model
from keras.engine import InputLayer
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.layers import Dense, Reshape, Dropout, Input, LSTM, Bidirectional
from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras import callbacks
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, LearningRateScheduler, ModelCheckpoint, Callback
from tensorflow.keras import backend as k

#custome libaries
from data_preprocessing_NetML import *
from data_preprocessing_IoT import IoT_data_common
from data_preprocessing_IoT import balance_data
from Autoencoder_IoT_model import build_iot_AE
from Autoencoder_NetML_model import build_NetML_AE

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

logdir = "/home/vibek/Anomanly_detection_packages/DNN_Package/logs_directory/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True)

params = {'dataset': 'IoT-23'}
result_path="/home/vibek/Anomanly_detection_packages/DNN_Package/"

print("Loading dataset IoT-23 .....\n")
train_data, test_data, train_labels,  test_labels = IoT_data_common(params)
print("train shape: ", train_data.shape)
print("test shape: ", test_data.shape)
print("train_label shape: ", train_labels.shape)
print("test_label shape: ", test_labels.shape)

print("Loading AutoEncoder model....\n")
autoencoder_1, encoder_1, autoencoder_2, encoder_2, autoencoder_3, encoder_3, sSAE, SAE_encoder = build_iot_AE()
print("value of SAE_encoder:\n", SAE_encoder.output)
print("value of SAE_encoder:\n", SAE_encoder.input)

#TimesereisGenerator
time_steps = 1
batch_size = 512
epochs = 100


print('Finding feature importances.....')

def find_importances(X_train, Y_train):
    model = ExtraTreesClassifier()
    model = model.fit(X_train, Y_train)
    
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]  # Top ranking features' indices
    return importances, indices, std


# Plot the feature importances of the forest
def plot_feature_importances(X_train, importances, indices, std, title):
   #tagy
#     for f in range(X_train.shape[1]):
#         print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    plt.figure(num=None, figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.title(title)
    width=5
    plt.bar(range(X_train.shape[1]), importances[indices],
          width=5, color="r", yerr=std[indices], align="center") #tagy 1.5 > .8
    plt.xticks(range(X_train.shape[1]), indices)
    #plt.axis('tight')
    plt.xlim([-1, X_train.shape[1]]) # -1 tagy
    plt.show()

# Neural network is classified with correct 'attack or not' labels
X_nn_train = np.concatenate((train_labels, train_data), axis=1)
nn_importances, nn_indices, nn_std = find_importances(X_nn_train,
                                                      train_labels)
plot_feature_importances(X_nn_train,
                        nn_importances, nn_indices, nn_std, title='Feature importances (IoT-23)')


#def build_model_DNN():
# 1. define the network
mlp0 = Dense(units=75, activation='relu')(SAE_encoder.output)
mlp0_drop = Dropout(0.3)(mlp0)

mlp1 = Dense(units=62, activation='relu')(mlp0_drop)
mlp_drop1 = Dropout(0.3)(mlp1)

mlp2 = Dense(units=50, activation='relu')(mlp_drop1)
mlp_drop2 = Dropout(0.3)(mlp2)

mlp3 = Dense(units=46, activation='relu')(mlp_drop2)
mlp_drop3 = Dropout(0.3)(mlp3)

mlp4 = Dense(units=1, activation='sigmoid')(mlp_drop3)

model = Model(SAE_encoder.input, mlp4)
model.summary()
plot_model(model,to_file='/home/vibek/Anomanly_detection_packages/DNN_Package/model_details_NetML_DNN.png',show_shapes=True)

# try using different optimizers and different optimizer configs
start = time.time()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print ("Compilation Time:", time.time() - start)
plot_losses = PlotLossesCallback()

#save model and the values
save_model = ModelCheckpoint(filepath="/home/vibek/Anomanly_detection_packages/DNN_Package/checkpoint-{epoch:02d}.hdf5", verbose=1, monitor='val_acc', save_best_only=True) 
csv_logger = CSVLogger('/home/vibek/Anomanly_detection_packages/DNN_Package/training_set_dnnanalysis_IoT_DNN.csv', separator=',', append=False)
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=100)
callbacks = [save_model, csv_logger, tensorboard_callback, early_stopping_monitor]

global_start_time = time.time()
print("Start Training...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_vald, y_vald), callbacks=callbacks)
model.save("/home/vibek/Anomanly_detection_packages/DNN_Package/best_model_iot_DNN.hdf5")
print("Done Training...")

#plot confusion matrix
def plot_confusion_matrix(cm, class_names):
  
    figure = pyplot.figure(figsize=(8, 8))
    pyplot.imshow(cm, interpolation='nearest', cmap=pyplot.cm.Blues)
    pyplot.title("Confusion matrix")
    pyplot.colorbar()
    tick_marks = np.arange(len(class_names))
    pyplot.xticks(tick_marks, class_names, rotation=45)
    pyplot.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        pyplot.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    pyplot.tight_layout()
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')
    pyplot.show()
    return figure

#plot ROC graph
def plot_roc_curve(Y_test, Y_pred, class_name, class_index, title='ROC'):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(class_name):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    #plt.figure()
    lw = 2
    pyplot.plot(fpr[class_index], tpr[class_index], color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[class_index])
    pyplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title(title)
    pyplot.legend(loc="lower right")
    pyplot.show()

#Save labels of each category 
def save_labels(predic, actual, result_path, phase, accur):
        labels = np.concatenate((predic, actual))

        
        if not os.path.exists(path=result_path):
            os.mkdir(path=result_path, exist_ok=True)

        np.save(file=os.path.join(result_path, '{}-DNN-Results-{}.npy'.format(phase, float('%.2f'%score))), arr=labels)

#label and category accuracy
def to_cat(y):
    y_tmp = np.ndarray(shape=(y.shape[0], 2), dtype=np.float32)
    for i in range(y.shape[0]):
        y_tmp[i, :] = np.array([1-y[i], y[i]])   # np.array([0,1]) if y[i] else np.array([1,0])
    return y_tmp


#LSTM model load
print('Loading IoT-23 model......')
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model_path = "/home/vibek/Anomanly_detection_packages/DNN_Package/Model_IoT/"
    model_name = model_path+"best_model_IoT_DNN.hdf5"
    model_IoT = load_model(model_name, compile=False)
    model_IoT.summary()

# try using different optimizers and different optimizer configs   
    start = time.time() 
    model_IoT.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

   #Print information of the results
print('Val loss and acc:\n')

 # Model for testing
print(model_IoT.evaluate(test_data, test_labels, batch_size=512, verbose=0))

np.savetxt("/home/vibek/Anomanly_detection_packages/DNN_Package/Training_Testing_result/model_evaluate_IoT_DNN.csv", model_IoT.evaluate(test_data, test_labels, verbose=1), delimiter=",")
#Print information of the results
print('Predict_result:\n')

# Model for prediction: Used to just return predicted values

test_pred_raw = model_IoT.predict(test_data, batch_size=512, verbose=0)
test_pred = np.argmax(test_pred_raw, axis=1)
test_label_original = np.argmax(test_labels, axis=1)
test_mae_loss = np.mean(np.power(test_data - test_pred[:, np.newaxis], 2), axis=1)
print('error_test:', test_mae_loss)
#print('test_label_original:', test_label_original)
#print('test_pred:', test_pred)

train_pred_raw = model_IoT.predict(train_data, batch_size=512, verbose=0)
train_pred = np.argmax(train_pred_raw, axis=1)
train_label_original = np.argmax(test_labels, axis=1)
train_mae_loss = np.mean(np.abs(train_pred[:, np.newaxis] - train_data), axis=1)
print('error_train:', train_mae_loss)


np.save('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_prediction_test_IoT_DNN.npy', test_pred)
np.save('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_original_test_IoT_DNN.npy', test_label_original)
np.save('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_prediction_train_IoT.npy', train_pred)
np.save('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_original_train_IoT.npy', train_label_original)
np.savetxt("/home/vibek/Anomanly_detection_packages/DNN_Package/Training_Testing_result/Predict_result_IoT_DNN.csv", test_pred, delimiter=",")

 # To encode string  labels into numbers
score = accuracy_score(test_label_original, test_pred)
print("Accuracy_score:", score)

print('Classification Report:\n')
print(classification_report(test_label_original, test_pred))

#Anomaly score
error_df = pd.DataFrame({'reconstruction_error': test_mae_loss,'true_class': test_label_original})
anomaly_error_df = error_df[error_df['true_class'] == 1]
print('Detection Result:', error_df)
print('Anomaly Score:', error_df.describe())

 #Visualize confusion matrix
print('\nConfusion Matrix:')
conf_matrix = confusion_matrix(test_label_original, test_pred)
print(conf_matrix)
figure = plot_confusion_matrix(conf_matrix, class_names=list(range(2))) 
save_labels(predic=test_pred, actual=test_label_original, result_path=result_path, phase='testing', accur=score)

# Log the confusion matrix as an image summary.
plot_roc_curve(to_cat(test_label_original), to_cat(test_pred), 2, 0, title='Receiver operating characteristic (attack_or_not = 0)')
plot_roc_curve(to_cat(test_label_original), to_cat(test_pred), 2, 1, title='Receiver operating characteristic (attack_or_not = 1)')

FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  # False Positive
FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)  # False Negative
TP = np.diag(conf_matrix)  # True Positive
TN = conf_matrix.sum() - (FP + FN + FP)  # True Negative

print('\nTPR:')  # True Positive Rate
# Portion of positive instances correctly predicted positive
print(TP / (TP + FN))
np.savetxt("/home/vibek/Anomanly_detection_packages/DNN_Package/Training_Testing_result/True_positive_rate_IoT_DNN.csv", TP / (TP + FN), delimiter=",")

print('\nFPR:')  # False Positive Rate
# Portion of negative instances wrongly predicted positive
print(FP / (FP + TN))
np.savetxt("/home/vibek/Anomanly_detection_packages/DNN_Package/Training_Testing_result/False_positive_rate_IoT_DNN.csv", FP / (FP + TN), delimiter=",")

# Cost Matrix as presented in Staudemeyer article
cost_matrix = [[0, 1, 2, 2, 2],
               [1, 0, 2, 2, 2],
               [2, 1, 0, 2, 2],
               [4, 2, 2, 0, 2],
               [4, 2, 2, 2, 0]]

cost_matrix = [[0, 1],
               [1, 0]]

tmp_matrix = np.zeros((2, 2))

for i in range(2):
    for j in range(2):
        tmp_matrix[i][j] = conf_matrix[i][j] * cost_matrix[i][j]

# The average cost is (total cost / total number of classifications)
print('\nCost:')
print(tmp_matrix.sum()/conf_matrix.sum())

print('\nAUC:')  # Average Under Curve
print(roc_auc_score(y_true=test_label_original, y_score=test_pred, average='macro'))

#pre-defined threshold
threshold=0.75

#Reconstruction error score
anomalies = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]

groups = error_df.groupby('true_class')
fig, ax = pyplot.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=2, linestyle='',
            label= "Anomaly" if name == 1 else "Normal", color= 'red' if name == 1 else 'black')
ax.legend()
pyplot.title("Reconstruction error score")
pyplot.ylabel("Reconstruction error")
pyplot.xlabel("Data point index")
pyplot.show();

#Plot anomalies graph to visualize the experimental results
s1 = np.load('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_original_test_IoT_DNN.npy')
s2 = np.load('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_prediction_test_IoT_DNN.npy')

test_pred_plot = s2 > 0.7

index_same = np.argwhere(s1 == test_pred_plot)
index_diff = np.argwhere(s1 != test_pred_plot)

normal = np.where(s1 == test_pred_plot)
anomaly = np.where(s1 != test_pred_plot)

dt = 1.0
t = np.arange(0, len(s1), dt)
s3 = np.ones(len(s1)) * 0.5
#print('The value of s3:', s3, t)

#plot the anomalies result figure
fig = pyplot.figure(1)
ax = fig.add_subplot(111)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

#classification results
ax.plot(t, s1, markersize=5, label='True Label', color='black', marker='*', linestyle='-')
ax.plot(t[index_same][:,0], test_pred_plot[index_same][:,0], markersize=3, label='correctly predicted', color='blue', marker='*', linestyle='')
ax.plot(t[index_diff][:,0], test_pred_plot[index_diff][:,0], markersize=3, label='wrongly predicted', color='red', marker='*', linestyle='')

#ax.plot(t, s3, 'r-')
#ax.set_ylim(-0.3, 1.5)
ax.set_xlabel('Number of Samples', font2)
ax.set_ylabel('Probability of Each Sample', font2)
ax.legend(loc='upper right', prop=font2)
pyplot.title('Classification Result', font1)
ax.grid(True)
pyplot.show()

# Draw the wrongly detected result values
fig2 = pyplot.figure(1)
ax1 = fig2.add_subplot(111)
count_line = np.zeros(len(s1))
index_low = 0
index_high = 0
for i, index in enumerate(index_diff):

    index_high = index[0]
    count_line[index_low:index_high] = i
    index_low = index_high
count_line[81918:] = 1506
ax1.plot(t, count_line)
pyplot.title('Cumulative Amount of Incorrect Detection', font1)
ax1.set_xlabel('Number of Samples', font2)
ax1.set_ylabel('Number of Incorrect Detection', font2)

pyplot.subplots_adjust(wspace=0., hspace =0.3)
pyplot.show()

# Create outlier detection plot
fig = pyplot.figure(1)
ax2 = fig.add_subplot(111)

ax2.scatter(normal, test_data[normal][:,0], c= 'blue', marker='*', label='Normal', s=1)
ax2.scatter(anomaly,test_data[anomaly][:,0], c= 'red', marker='*', label='Anomaly', s=5)
pyplot.title('Anomaly Detection')
pyplot.legend(loc=2)
pyplot.show()

