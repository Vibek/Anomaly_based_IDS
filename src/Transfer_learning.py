from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

#Python libraries
from datetime import datetime
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
import seaborn as sns
import model_evaluation_utils as meu
from IPython.display import clear_output

##Sklearn libraries
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, auc, recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.metrics import make_scorer, auc
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedKFold,cross_val_predict
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
from shap import TreeExplainer, KernelExplainer, DeepExplainer, force_plot, GradientExplainer, LinearExplainer
from deeplift.visualization import viz_sequence

#keras libraries
from keras.models import load_model, model_from_json, Model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from livelossplot.keras import PlotLossesCallback
from keras.engine import InputLayer
from keras.optimizers import Adam, SGD
from keras.utils import plot_model, to_categorical
from keras.layers import concatenate, ZeroPadding2D, Dense, Reshape, Dropout, Input, LSTM, Bidirectional, Flatten, GlobalAveragePooling2D
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras import callbacks
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, LearningRateScheduler, ModelCheckpoint, Callback
from tensorflow.keras import backend as K

#custome libaries
from data_preprocessing_unsw import unsw_data_common
from data_preprocessing_CICDS import cicids_data_common
from data_preprocessing_IoT import IoT_data_common
from data_preprocessing_IoT import balance_data
from Autoencoder_unsw_model import build_unsw_AE
from Autoencoder_CICDS_model import build_cicds_AE
from Autoencoder_IoT_model import build_iot_AE

#TimesereisGenerator
time_steps = 1
batch_size = 512
epochs = 50
num_class = 2
np.random.seed(0)
value=1.5
width=0.75

logdir = "/home/vibek/Anomanly_detection_packages/DNN_Package/logs_directory/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True)

params = {'dataset': 'IoT-23'}
result_path="/home/vibek/Anomanly_detection_packages/DNN_Package/"

#loading the data from data preprocessing
print("Loading dataset UNSW-2015_data.....\n")
train_data_unsw, train_labels_unsw, test_data_unsw, test_labels_unsw = unsw_data_common(params)
print("Loading dataset CICIDS-2017_data.....\n")
train_data_cicds, train_labels_cicds, test_data_cicds, test_labels_cicds = cicids_data_common(params)
print("Loading dataset IoT-23_data.....\n")
train_data_iot, train_labels_iot, test_data_iot, test_labels_iot = IoT_data_common(params)

print("Loading unsw AutoEncoder model....\n")
autoencoder_unsw_1, encoder_unsw_1, autoencoder_unsw_2, encoder_unsw_2, autoencoder_unsw_3, encoder_unsw_3, sSAE_unsw, SAE_encoder_unsw = build_unsw_AE()
print("value of SAE_encoder:\n", SAE_encoder_unsw.output)
print("value of SAE_encoder:\n", SAE_encoder_unsw.input)

print("Loading cicds AutoEncoder model....\n")
autoencoder_cicds_1, encoder_cicds_1, autoencoder_cicds_2, encoder_cicds_2, autoencoder_cicds_3, encoder_cicds_3, sSAE_cicds, SAE_encoder_cicds = build_cicds_AE()
print("value of SAE_encoder:\n", SAE_encoder_cicds.output)
print("value of SAE_encoder:\n", SAE_encoder_cicds.input)

print("Loading IoT AutoEncoder model....\n")
autoencoder_iot_1, encoder_iot_1, autoencoder_iot_2, encoder_iot_2, autoencoder_iot_3, encoder_iot_3, sSAE_iot, SAE_encoder_iot = build_iot_AE()
print("value of SAE_encoder:\n", SAE_encoder_iot.output)
print("value of SAE_encoder:\n", SAE_encoder_iot.input)

test_label_iot_original = np.argmax(test_labels_iot, axis=1)
train_label_iot_original = np.argmax(test_data_iot, axis=1)

X_train = train_data_iot.astype('float32')
print(X_train.shape)
X_val = test_data_iot.astype('float32')
print(X_val.shape)

class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
    	self.logs.append(timer()-self.starttime)

'''
def build_model_UNSW():

	kfold = KFold(n_splits=5, shuffle=True, random_state=42)
	cvscores = []
	Fold = 1
	for train, val in kfold.split(train_data_unsw, test_data_unsw):
		gc.collect()
		print ('Fold: ',Fold)

		X_train = train_data_unsw[train]
		X_val = train_data_unsw[val]
		X_train = X_train.astype('float32')
		X_val = X_val.astype('float32')
		y_train = test_data_unsw[train]
		y_val = test_data_unsw[val]

		# 1. define the network
		mlp0 = Dense(units=52, activation='relu')(SAE_encoder_unsw.output)
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

		model = Model(SAE_encoder_unsw.input, mlp4)
	
		plot_model(model,to_file='/home/vibek/Anomanly_detection_packages/DNN_Package/models/model_details_unsw.png',show_shapes=True)

		start = time.time()
		model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
		model.summary()
		print ("Compilation Time:", time.time() - start)

		cb = TimingCallback()
		early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=50)
		callbacks = [tensorboard_callback, early_stopping_monitor, cb]

		# Start multiple model training with the batch size
		global_start_time = time.time()
		print("Start Training...")
		model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks)
		print(sum(cb.logs))

		filename = '/home/vibek/Anomanly_detection_packages/DNN_Package/Model_IoT/model_1.h5'
		model.save(filename)
		print('>Saved %s' % filename)

		# evaluate the model
		scores = model.evaluate(X_val, y_val, verbose=0)
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)
		Fold = Fold +1
		print("Done Training...")

		#Explainability of the model
		explainer = GradientExplainer(model_f, data=X_train[:100])
		print(explainer)
		shap_values = explainer.shap_values(X_val[:100])
		shap_values = shap_values[0]
		pd.DataFrame(shap_values).head()
		print('Expected Value:', explainer.expected_value[0])

		shap.summary_plot(shap_values, X_val[:10])
		shap.summary_plot(shap_values, X_val[:10], plot_type="bar")

	print("%s: %.2f%%" % ("Mean Accuracy: ",np.mean(cvscores)))
	print("%s: %.2f%%" % ("Standard Deviation: +/-", np.std(cvscores)))

	return model

def build_model_CICDS():

	kfold = KFold(n_splits=5, shuffle=True, random_state=42)
	cvscores = []
	Fold = 1
	for train, val in kfold.split(train_data_cicds, test_data_cicds):
		gc.collect()
		print ('Fold: ',Fold)

		X_train = train_data_cicds[train]
		X_val = train_data_cicds[val]
		X_train = X_train.astype('float32')
		X_val = X_val.astype('float32')
		y_train = test_data_cicds[train]
		y_val = test_data_cicds[val]

		# 1. define the network
		mlp0 = Dense(units=52, activation='relu')(SAE_encoder_cicds.output)
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

		model = Model(SAE_encoder_cicds.input, mlp4)
	
		plot_model(model,to_file='/home/vibek/Anomanly_detection_packages/DNN_Package/models/model_details_cicds.png',show_shapes=True)

		start = time.time()
		model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
		model.summary()
		print ("Compilation Time:", time.time() - start)

		cb = TimingCallback()
		early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=50)
		callbacks = [tensorboard_callback, early_stopping_monitor, cb]

		# Start multiple model training with the batch size
		global_start_time = time.time()
		print("Start Training...")
		model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks)
		print(sum(cb.logs))

		filename = '/home/vibek/Anomanly_detection_packages/DNN_Package/Model_IoT//model_2.h5'
		model.save(filename)
		print('>Saved %s' % filename)

		# evaluate the model
		scores = model.evaluate(X_val, y_val, verbose=0)
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)
		Fold = Fold +1
		print("Done Training...")

		#Explainability of the model
		explainer = GradientExplainer(model_f, data=X_train[:100])
		print(explainer)
		shap_values = explainer.shap_values(X_val[:100])
		shap_values = shap_values[0]
		pd.DataFrame(shap_values).head()
		print('Expected Value:', explainer.expected_value[0])

		shap.summary_plot(shap_values, X_val[:10])
		shap.summary_plot(shap_values, X_val[:10], plot_type="bar")

	print("%s: %.2f%%" % ("Mean Accuracy: ",np.mean(cvscores)))
	print("%s: %.2f%%" % ("Standard Deviation: +/-", np.std(cvscores)))

	return model

def build_model_Mawilab():

	kfold = KFold(n_splits=5, shuffle=True, random_state=42)
	cvscores = []
	Fold = 1
	for train, val in kfold.split(train_data_iot, test_data_iot):
		gc.collect()
		print ('Fold: ',Fold)

		X_train = train_data_iot[train]
		X_val = train_data_iot[val]
		X_train = X_train.astype('float32')
		X_val = X_val.astype('float32')
		y_train = test_data_iot[train]
		y_val = test_data_iot[val]

		# 1. define the network
		mlp0 = Dense(units=52, activation='relu')(SAE_encoder_iot.output)
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

		model = Model(SAE_encoder_iot.input, mlp4)
	
		plot_model(model,to_file='/home/vibek/Anomanly_detection_packages/DNN_Package/models/model_details_mawilab.png',show_shapes=True)

		start = time.time()
		model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
		model.summary()
		print ("Compilation Time:", time.time() - start)

		cb = TimingCallback()
		early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=50)
		callbacks = [tensorboard_callback, early_stopping_monitor, cb]

		# Start multiple model training with the batch size
		global_start_time = time.time()
		print("Start Training...")
		model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks)
		print(sum(cb.logs))

		filename = '/home/vibek/Anomanly_detection_packages/DNN_Package/Model_IoT//model_3.h5'
		model.save(filename)
		print('>Saved %s' % filename)

		# evaluate the model
		scores = model.evaluate(X_val, y_val, verbose=0)
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)
		Fold = Fold +1
		print("Done Training...")

		#Explainability of the model
		explainer = GradientExplainer(model_f, data=X_train[:100])
		print(explainer)
		shap_values = explainer.shap_values(X_val[:100])
		shap_values = shap_values[0]
		pd.DataFrame(shap_values).head()
		print('Expected Value:', explainer.expected_value[0])

		shap.summary_plot(shap_values, X_val[:10])
		shap.summary_plot(shap_values, X_val[:10], plot_type="bar")

	print("%s: %.2f%%" % ("Mean Accuracy: ",np.mean(cvscores)))
	print("%s: %.2f%%" % ("Standard Deviation: +/-", np.std(cvscores)))

	return model

model = []
model.append(build_model_UNSW())
model.append(build_model_CICDS())
model.append(build_model_Mawilab())
'''
# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = '/home/vibek/Anomanly_detection_packages/DNN_Package/Model_IoT//model_' + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models

# load all models
n_members = 3
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# define stacked model from multiple member input models
def define_stacked_model(members):
	# update all layers in all models to not be trainable
	for i in range(len(members)):
		model = members[i]
		print('model information',model)
		for layer in model.layers:
			# make not trainable
			layer.trainable = False
			# rename to avoid 'unique layer name' issue
			layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
	# define multi-headed input
	ensemble_inputs = [model.input for model in members]
	print('model input:', ensemble_inputs)
	# concatenate merge output from each model
	ensemble_outputs = [model.output for model in members]
	print('model ouput:', ensemble_outputs)
	merge = concatenate(ensemble_outputs)
	hidden = Dense(15, activation='relu')(merge)
	hidden_drop = Dropout(0.1)(hidden)
	hidden_batch = BatchNormalization()(hidden_drop)
	hidden1 = Dense(10, activation='relu')(hidden_batch)
	hidden_drop1 = Dropout(0.1)(hidden1)
	hidden_batch1 = BatchNormalization()(hidden_drop1)
	output = Dense(2, activation='sigmoid')(hidden_batch1)
	model = Model(inputs=ensemble_inputs, outputs=output)
	# plot graph of ensemble
	plot_model(model, show_shapes=True, to_file='/home/vibek/Anomanly_detection_packages/DNN_Package/models/integrated_stcaked_model_graph.png')

	print("[INFO] compiling model...")
	opt = SGD(lr=1e-4, momentum=0.7)
	# compile
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()
	return model

# fit a stacked model
def fit_integrated_stacked_model(model, inputX, inputy):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# encode output data
	#inputy_enc = to_categorical(inputy)
	cb = TimingCallback()
	# fit model
	model.fit(X, inputy, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[cb])
	print('Total Training time (seconds):',sum(cb.logs))

	filename = '/home/vibek/Anomanly_detection_packages/DNN_Package/Model_IoT/stacked_model.h5'
	model.save(filename)
	print('>Saved %s' % filename)

# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# make prediction
	return model.predict(X, verbose=0)

def get_occuracy(real_labels, predicted_labels, fltr):
	real_label_count = 0.0
	predicted_label_count = 0.0

	for real_label in real_labels:
		if real_label == fltr:
			real_label_count += 1

			for predicted_label in predicted_labels:
				if predicted_label == fltr:
					predicted_label_count += 1
					print ("Real number of attacks: " + str(real_label_count))
					print ("Predicted number of attacks: " + str(predicted_label_count))
					precision = predicted_label_count * 100 / real_label_count
					return precision

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
	#model = svm.SVC(kernel='rbf', probability=True)
	#model.fit(stackedX, input_data_y)
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

# fit stacked model using the ensemble
model = fit_stacked_model(members, train_data_iot, train_label_iot_original)
# evaluate model on test set
y_pred = stacked_prediction(members, model, train_labels_iot)
print(y_pred)
#accyracy score of stacked model
stack_acc = accuracy_score(test_label_iot_original, y_pred)
print('Trabsfer learning Accuracy (feature extraction): %.3f' % stack_acc)

class_labels = [0, 1]
meu.display_model_performance_metrics(true_labels=test_label_iot_original, 
                                      predicted_labels=y_pred, 
                                      classes=class_labels)

_= xai.metrics_plot(test_label_iot_original, y_pred)

_ = xai.roc_plot(test_label_iot_original, y_pred)

d = xai.smile_imbalance(test_label_iot_original, y_pred)

d[["correct", "incorrect"]].sum().plot.bar()

# define ensemble model
stacked_model = define_stacked_model(members)
print(stacked_model)
# fit stacked model on test dataset
fit_integrated_stacked_model(stacked_model, train_data_iot, test_data_iot)

# define filename for this ensemble
filename = '/home/vibek/Anomanly_detection_packages/DNN_Package/Model_IoT/stacked_model.h5'
# load model from file
model_f = load_model(filename)
#model.load_weights(filename)
print('>loaded stacked_model %s' % filename)

# make predictions and evaluate
yhat = predict_stacked_model(stacked_model, train_labels_iot)
yhat = np.argmax(yhat, axis=1)
print(yhat)
acc = accuracy_score(test_label_iot_original, yhat)
print('Trabsfer learning Accuracy (fine tuning): %.3f' % acc)

_= xai.metrics_plot(test_label_iot_original, yhat)

_ = xai.roc_plot(test_label_iot_original, yhat)

d = xai.smile_imbalance(test_label_iot_original, yhat)

d[["correct", "incorrect"]].sum().plot.bar()
      
class_labels = [0, 1]
meu.display_model_performance_metrics(true_labels=test_label_iot_original, 
                                      predicted_labels=yhat, 
                                      classes=class_labels)

print ("The precision of the Transfer learning classifier is: " + str(get_occuracy(test_label_iot_original,yhat, 1)) + "%")

# Calculate Normal distribution plot
RMSE_te = np.sqrt(np.mean((test_label_iot_original.ravel()- yhat.ravel())**2))
print('RMSE_stacked_esemble', RMSE_te)

#Anomaly score
error_df = pd.DataFrame({'reconstruction_error': RMSE_te,'true_class': test_label_iot_original})
anomaly_error_df = error_df[error_df['true_class'] == 1]

np.save('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_prediction_test.npy', yhat)
np.save('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_prediction_test_f.npy', y_pred)
np.save('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_original_test.npy', test_label_iot_original)

#Plot anomalies graph to visualize the experimental results
s1 = np.load('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_original_test.npy')
s2 = np.load('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_prediction_test.npy')
s3 = np.load('/home/vibek/Anomanly_detection_packages/DNN_Package/plot_prediction_test_f.npy')

test_pred_plot = s2 > 0.85

test_pred_plot_f = s3 > 0.85

normal = np.where(s1 == test_pred_plot)
anomaly = np.where(s1 != test_pred_plot)

normal_f = np.where(s1 == test_pred_plot_f)
anomaly_f = np.where(s1 != test_pred_plot_f)

# Create outlier detection plot
fig = pyplot.figure(1)
ax2 = fig.add_subplot(111)

ax2.scatter(normal, train_labels_iot[normal][:,0], c= 'blue', marker='*', label='Data', s=1)
ax2.scatter(anomaly,train_labels_iot[anomaly][:,0], c= 'red', marker='*', label='Anomaly', s=5)
pyplot.title('Anomaly Detection Fine Tuning')
pyplot.legend(loc=2)
pyplot.show()

# Create outlier detection plot
fig = pyplot.figure(1)
ax2 = fig.add_subplot(111)

ax2.scatter(normal_f, train_labels_iot[normal_f][:,0], c= 'blue', marker='*', label='Data', s=1)
ax2.scatter(anomaly_f,train_labels_iot[anomaly_f][:,0], c= 'red', marker='*', label='Anomaly', s=5)
pyplot.title('Anomaly Detection Feature Extraction')
pyplot.legend(loc=2)
pyplot.show()

#plot classifier accuracy 
T = [92.162, 57.539]
pyplot.figure()
pyplot.plot(T, 'ro-')                                                                                                                              
pyplot.xticks(range(3), ['feature_extraction', 'fine tuning'])        
pyplot.ylabel('Training Time (ms)'); pyplot.xlabel('Classifier'); pyplot.title('Training time graph');
pyplot.show()
