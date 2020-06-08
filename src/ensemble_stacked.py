from __future__ import print_function

#Python libraries
from datetime import datetime
import seaborn as sns
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from numpy import dstack
from scipy import stats
from matplotlib import pyplot
import io
import itertools
import pickle
import shutil
import time
from os import makedirs
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
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn import preprocessing
import scikitplot as skplt
import h5py
import gc
import shap
import xai
import xai.data

#mlxtend (XAI) libraries
from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions
from mlxtend.classifier import StackingClassifier
from timeit import default_timer as timer
from shap import TreeExplainer, KernelExplainer, DeepExplainer, force_plot, GradientExplainer
from deeplift.visualization import viz_sequence

#keras libraries
from keras.models import load_model
from keras.models import model_from_json
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from livelossplot.keras import PlotLossesCallback
from keras.models import Model
from keras.engine import InputLayer
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
from keras.layers import concatenate, ZeroPadding2D
from keras.layers import Dense, Reshape, Dropout, Input, LSTM, Bidirectional, Flatten
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras import callbacks
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, LearningRateScheduler, ModelCheckpoint, Callback
from tensorflow.keras import backend as K

#custome libaries
from data_preprocessing_NetML import *
from data_preprocessing_LITNET import LITNET_data
from data_preprocessing_IoT import IoT_data_common
from data_preprocessing_IoT import balance_data
from Autoencoder_IoT_model import build_iot_AE
from Autoencoder_NetML_model import build_NetML_AE
from Autoencoder_LITNET_model import build_LITNET_AE

#TimesereisGenerator
time_steps = 1
batch_size = 512
epochs = 50
num_class = 2 # 1 (for NetML)
np.random.seed(0)
value=1.5
width=0.75

logdir = "/home/vibek/Anomanly_detection_packages/DNN_Package/logs_directory/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True)


# Create folder for the results
time_ = t.strftime("%Y%m%d-%H%M%S")

save_dir = os.getcwd() + '/results/' + time_
os.makedirs(save_dir)


params = {'dataset': 'IoT-23'}
result_path="/home/vibek/Anomanly_detection_packages/DNN_Package/"
'''
###LITNET-2020 dataset####
print("Loading dataset LITNET-2020.....\n")
train_data, train_labels, test_data, test_labels = LITNET_data(params)
print("LITNET_data train shape: ", train_data.shape)
print("LITNET_data train_label shape: ", test_data.shape)
print("LITNET_data validation shape: ", train_labels.shape)
print("LITNET_data Validation_label shape: ", test_labels.shape)
test_label_original = np.argmax(test_labels, axis=1)
train_label_original = np.argmax(test_data, axis=1)


###IoT-23 dataset####
print("Loading dataset IoT-23.....\n")
train_data_i, train_labels_i, test_data_i, test_labels_i = IoT_data_common(params)
print("train shape: ", train_data_i.shape)
print("test shape: ", test_data_i.shape)
print("train_label shape: ", train_labels_i.shape)
print("test_label shape: ", test_labels_i.shape)
#test_label_original = np.argmax(test_labels_i, axis=1)
#train_label_original = np.argmax(test_data_i, axis=1)

'''
###NetML dataset####
print("\n Loading dataset NetML-2020......\n")
dataset = "/home/vibek/Anomanly_detection_packages/NetML-2020/data/NetML" 
anno = "top" # or "mid" or "fine"
submit = "both" # or "test-std" or "test-challenge"

# Assign variables
training_set = dataset+"/2_training_set"
training_anno_file = dataset+"/2_training_annotations/2_training_anno_"+anno+".json.gz"
test_set = dataset+"/1_test-std_set"
challenge_set = dataset+"/0_test-challenge_set"


# Get training data in np.array format
Xtrain, ytrain, class_label_pair, Xtrain_ids = get_training_data(training_set, training_anno_file)
print('Training class:\n', class_label_pair)

Xtest, ids = get_testing_data(test_set)

# Split validation set from training data
XX_train, X_vald, yy_train, y_vald = train_test_split(Xtrain, ytrain,
                                                test_size=0.2, 
                                                random_state=42,
                                                stratify=ytrain)

print("NetML train shape: ", XX_train.shape)
print("NetML train_label shape: ", yy_train.shape)
print("NetML validation shape: ", X_vald.shape)
print("NetML Validation_label shape: ", y_vald.shape)
print("NetML test shape: ", Xtest.shape)

# Preprocess the data
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(XX_train)
X_val_scaled = scaler.transform(X_vald)
X_test_scaled = scaler.transform(Xtest)

#selected_feat= XX_train.columns[(sel.get_support())]

print("Loading AutoEncoder model....\n")
autoencoder_1, encoder_1, autoencoder_2, encoder_2, autoencoder_3, encoder_3, sSAE, SAE_encoder = build_LITNET_AE() #build_NetML_AE(), build_IoT_AE()
print("value of SAE_encoder:\n", SAE_encoder.output)
print("value of SAE_encoder:\n", SAE_encoder.input)

class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(timer()-self.starttime)
'''
def build_model_DNN():
# 1. define the network
	mlp0 = Dense(units=52, activation='relu')(SAE_encoder.output)
	mlp0_drop = Dropout(0.3)(mlp0)

	mlp1 = Dense(units=34, activation='relu')(mlp0_drop)
	mlp_drop1 = Dropout(0.3)(mlp1)

	mlp2 = Dense(units=26, activation='relu')(mlp_drop1)
	mlp_drop2 = Dropout(0.3)(mlp2)

	mlp3 = Dense(units=20, activation='relu')(mlp_drop2)
	mlp_drop3 = Dropout(0.3)(mlp3)

	mlp4 = Dense(units=16, activation='relu')(mlp_drop3)
	mlp_drop4 = Dropout(0.3)(mlp4)

	mlp4 = Dense(units=num_class, activation='sigmoid')(mlp_drop4)

	model = Model(SAE_encoder.input, mlp4)
	
	#plot_model(model,to_file='/home/vibek/Anomanly_detection_packages/DNN_Package/model_details_unsw.png',show_shapes=True)

	start = time.time()
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.summary()
	print ("Compilation Time:", time.time() - start)

	return model

#tf.compat.v1.disable_eager_execution()
def build_model_LSTM():
# 1. define the network
    mlp0 = Dense(units=32, activation='relu')(SAE_encoder.output)

    lstm_reshape = Reshape((1, 32))(mlp0)

    lstm1 = LSTM(units=24, activation='tanh', return_sequences=True)(lstm_reshape)

    lstm1_reshape = Reshape((1, 24))(lstm1)

    lstm2 = LSTM(units=16, activation='tanh', return_sequences=True)(lstm1_reshape)

    lstm2_reshape = Reshape((1, 16))(lstm2)

    lstm3 = LSTM(units=10, activation='tanh', return_sequences=True)(lstm2_reshape)

    lstm3_reshape = Reshape((1, 10))(lstm3)

    lstm4 = LSTM(units=6, activation='tanh', return_sequences=True)(lstm3_reshape)

    lstm4_reshape = Reshape((1, 6))(lstm4)

    lstm5 = LSTM(units=num_class, activation='sigmoid')(lstm4_reshape)

    model = Model(SAE_encoder.input, lstm5)
    
    #plot_model(model,to_file='/home/vibek/Anomanly_detection_packages/DNN_Package/model_details_cicds.png',show_shapes=True)

    start = time.time()
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    print("Compilation Time:", time.time() - start)

    return model


# create directory for models
#makedirs('/home/vibek/Anomanly_detection_packages/DNN_Package/models_LITNET')

model = []
model.append(build_model_DNN())
model.append(build_model_LSTM())


kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cvscores = []
Fold = 1
for train, val in kfold.split(train_data, test_data): #(X_train_scaled, yy_train)-NetML,  (train_data_i, test_data_i)-IoT-23
    gc.collect()
    #K.clear_session()
    print ('Fold: ',Fold)
    
    X_train = train_data[train]
    X_val = train_data[val]
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    y_train = test_data[train]
    y_val = test_data[val]

    cb = TimingCallback()
    save_model = ModelCheckpoint(filepath="/home/vibek/Anomanly_detection_packages/DNN_Package/checkpoint-{epoch:02d}.hdf5", verbose=1, monitor='val_acc', save_best_only=True) 
    csv_logger = CSVLogger('/home/vibek/Anomanly_detection_packages/DNN_Package/training_set_dnnanalysis_NetML_DNN.csv', separator=',', append=False)
    early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=50)
    callbacks = [save_model, csv_logger, tensorboard_callback, early_stopping_monitor, cb]

    # Start multiple model training with the batch size
    global_start_time = time.time()
    print("Start Training...")
    models = []
    for i in range(len(model)):
    	model[i].fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks)
    	models.append(model[i])
    	print(sum(cb.logs))

    	# save model
    	filename = '/home/vibek/Anomanly_detection_packages/DNN_Package/models_LITNET/model_' + str(i + 1) + '.h5'
    	model[i].save(filename)
    	print('>Saved %s' % filename)
    	
    	#Explainability of the model
    	explainer = GradientExplainer(model[i], data=X_train[:100])
    	print(explainer)
    	shap_values = explainer.shap_values(X_val[:10])
    	shap_values = shap_values[0]
    	pd.DataFrame(shap_values).head()
    	#print('Expected Value:', explainer.expected_value[0])

    	shap.summary_plot(shap_values, X_val[:10])
    	shap.summary_plot(shap_values, X_val[:10], plot_type="bar")
		
    	# evaluate the model
    	scores = model[i].evaluate(X_val, y_val, verbose=0)
    	print("%s: %.2f%%" % (model[i].metrics_names[1], scores[1]*100))
    	cvscores.append(scores[1] * 100)
    	Fold = Fold +1
    print("Done Training...")
print("%s: %.2f%%" % ("Mean Accuracy: ",np.mean(cvscores)))
print("%s: %.2f%%" % ("Standard Deviation: +/-", np.std(cvscores)))
'''
# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = '/home/vibek/Anomanly_detection_packages/DNN_Package/models_NetML/model_' + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models

# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, input_data_x):
	stackX = None
	for model in members:
		# make prediction
		y_pred = model.predict(input_data_x, verbose=0)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = y_pred
		else:
			stackX = dstack((stackX, y_pred))
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX
 
# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, input_data_x, input_data_y):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, input_data_x)
	#print(stackedX)
	#clf = svm.SVC(verbose=True)
	#clf.fit(stackedX, input_data_y)
	# fit standalone model
	model = LogisticRegression()
	model.fit(stackedX, input_data_y)
	return model
 
# make a prediction with the stacked model
def stacked_prediction(members, model, input_data_x):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, input_data_x)
	# make a prediction
	y_pred = model.predict(stackedX)
	return y_pred

# define stacked model from multiple member input models
def define_stacked_model(members):
	# update all layers in all models to not be trainable
	for i in range(len(members)):
		model = members[i]
		for layer in model.layers:
			# make not trainable
			layer.trainable = False
			# rename to avoid 'unique layer name' issue
			layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
	# define multi-headed input
	ensemble_inputs = [model.input for model in members]
	# concatenate merge output from each model
	ensemble_outputs = [model.output for model in members]
	merge = concatenate(ensemble_outputs)
	hidden = Dense(10, activation='relu')(merge)
	hidden_drop = Dropout(0.3)(hidden)
	output = Dense(num_class, activation='sigmoid')(hidden_drop) 
	model = Model(inputs=ensemble_inputs, outputs=output)
	# plot graph of ensemble
	plot_model(model, show_shapes=True, to_file='/home/vibek/Anomanly_detection_packages/DNN_Package/models_NetML/integrated_stcaked_model_graph.png')
	# compile
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# fit a stacked model
def fit_integrated_stacked_model(model, inputX, inputy):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# encode output data
	#inputy_enc = to_categorical(inputy)
	cb = TimingCallback()
	# fit model
	model.fit(X, inputy, batch_size=batch_size, epochs=100, verbose=0, callbacks=[cb])
	print(sum(cb.logs))

# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# make prediction
	return model.predict(X, verbose=0)


# Define function to submit the final results (Malware or bening)
def result_CSV(model, test_set, scaler, class_label_pair, filepath):
    #Xtest, ids = get_submission_data(test_set)
    #X_test_scaled = scaler.transform(Xtest)
    print("Predicting on {} ...".format(test_set.split('/')[-1]))
    predictions = model.predict(X_test_scaled)
    make_result_submission(predictions, ids, class_label_pair, filepath)

# load all models
n_members = 2
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# Predict labels with models
labels = []
scores = []
cvscores = []

for m in members:
	_, acc = m.evaluate(X_val_scaled, y_vald, batch_size=512, verbose=0)
	print('Model Accuracy: %.3f' % acc)

	predicts = np.argmax(m.predict(X_val_scaled), axis=1)
	print('predicts\n', predicts)
	labels.append(predicts)

	print('Classification Report:\n')
	print(classification_report(y_vald, predicts))

	# To encode string  labels into numbers
	score = accuracy_score(y_vald, predicts)
	print('\nscore\n', score)
	scores.append(score)
	
# summarize the distribution of scores
print('Scores Mean: %.3f, Standard Deviation: %.3f' % (np.mean(scores), np.std(scores)))
    
# Ensemble with voting
labels = np.array(labels)
outcomes_mean = np.mean(labels)
print("Average model accuracy_score:\n", outcomes_mean)

# sum across ensembles
summed = np.argmax(np.sum(labels, axis=0), axis=0)

# fit stacked model using the ensemble
model = fit_stacked_model(members, X_val_scaled, y_vald)
# evaluate model on test set
y_pred = stacked_prediction(members, model, X_val_scaled)
print(y_pred)
#accyracy score of stacked model
stack_acc = accuracy_score(y_vald, y_pred)
print('Stacked Test Accuracy: %.3f' % stack_acc)

print('Classification Report:\n')
print(classification_report(y_vald, y_pred))

# Calculate Normal distribution plot
RMSE_tr = np.sqrt(np.mean((y_vald.ravel()- outcomes_mean.ravel())**2))
RMSE_te = np.sqrt(np.mean((y_vald.ravel()- y_pred.ravel())**2))
print('RMSE_average_model', RMSE_tr)
print('RMSE_stacked_esemble', RMSE_te)

# Draw a nested barplot to show the training time of each model
df = pd.DataFrame([['UNSW-NB15','DNN',463.118],['UNSW-NB15','LSTM',970.600],['UNSW-NB15','Stacked',298.169],['CICIDS-2017','DNN',723.917],
                   ['CICIDS-2017','LSTM',1570.361],['CICIDS-2017','Stacked',429.609], ['IoT-23','DNN',80.95], ['IoT-23','LSTM',121.602], ['IoT-23','Stacked',30.63]],columns=['dataset','classifier','Training time'])

df.pivot("classifier", "dataset", "Training time").plot(kind='bar')

plt.show()
'''
plt.figure()
plot_learning_curves(train_data, test_data, train_labels, test_labels, model, print_model=False, style='ggplot')
plt.show()
'''
# Make submission of result to CSV format
if submit == "test-std" or submit == "both":
    result_CSV(model, test_set, scaler, class_label_pair, save_dir+"/submission_test-std.csv")
if submit == "test-challenge" or submit == "both":
    result_CSV(model, challenge_set, scaler, class_label_pair, save_dir+"/submission_test-challenge.csv")
'''
# define ensemble model (Integrated stack model)
stacked_model = define_stacked_model(members)
print(stacked_model)
# fit stacked model on test dataset
fit_integrated_stacked_model(stacked_model, train_data, train_label_original)
# make predictions and evaluate
yhat = predict_stacked_model(stacked_model, train_labels)
yhat = np.argmax(yhat, axis=1)
print(yhat)
acc = accuracy_score(test_label_original, yhat)
print('Integrated Stacked Test Accuracy: %.3f' % acc)
'''
np.save('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_prediction_test.npy', y_pred)
np.save('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_original_test.npy', y_vald)

#Plot anomalies graph to visualize the experimental results
s1 = np.load('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_original_test.npy')
s2 = np.load('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_prediction_test.npy')

test_pred_plot = s2 > 0.7

normal = np.where(s1 == test_pred_plot)
anomaly = np.where(s1 != test_pred_plot)

# Create outlier detection plot
fig = pyplot.figure(1)
ax2 = fig.add_subplot(111)

ax2.scatter(normal, X_val_scaled[normal][:,0], c= 'blue', marker='*', label='Normal', s=1)
ax2.scatter(anomaly, X_val_scaled[anomaly][:,0], c= 'red', marker='*', label='Anomaly', s=5)
pyplot.title('Anomaly Detection')
pyplot.legend(loc=2)
pyplot.show()
