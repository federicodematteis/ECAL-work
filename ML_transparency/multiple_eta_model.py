#automate data preparation with few functions 
#baseline setup
from ast import In
from cmath import exp
from itertools import tee
from random import sample
from tabnanny import check
from tkinter.ttk import LabelFrame
from importlib_metadata import Sectioned
import matplotlib.pyplot as plt
#import ROOT
import matplotlib as mpl
import pandas as pd
import numpy as np

#libraries for general data analysis 
from pprint import pprint
from collections import namedtuple
import datetime
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.backend import get_session

#libraries for DNN
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, LeakyReLU, Add, Concatenate, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM  #might be usefull for further works
from tensorflow.keras.utils  import plot_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import seaborn as sns
from tensorflow.keras import callbacks

from keras.regularizers import l2
from keras.regularizers import l1



from tensorflow.keras.models import load_model
#from keras.models import load_model

data_folder = ('/home/federico/root/root-6.24.06-install')
#data_folder = ('/gwpool/users/fdematteis') #CERN server
#dictionary_train for npy files containing rough transparency data



#--------------------------------Machine Learning ─=≡Σ(([ ⊐•̀⌂•́]⊐-----------------------------------------------

#chiamare le funzioni contenute negli script di data-preparation: Pre_Processing_test & Pre_Processing_train
#le funzioni in questione restituiscono diverse variabili in output, necessarie per questo script.


#custom loss function for training the model with sample weighting 

def delta_train(transp_training,transp_predicted_train_loss):
    loss = K.square(transp_training-transp_predicted_train_loss)
    loss=loss*weights_train
    loss=K.sum(loss, axis=1)
    return loss

#test metrics with sample weighting for validation dataset (not usefull) 
def delta_test(transp_test_final,transp_predicted_test_loss):
    loss = K.square(transp_test_final-transp_predicted_test_loss)
    loss=loss*weights_validation
    loss=K.sum(loss, axis=1)
    return loss

from math import exp
import neptune.new as neptune

def lr_exp_decay(epoch):
    k = 0.1
    initial_learning_rate = 0.00001  
    return initial_learning_rate * exp(-k*epoch)


lrate = callbacks.LearningRateScheduler(lr_exp_decay)


from tensorflow.keras import activations
def clip(x, max_value=1.5):
    return activations.relu(x, max_value=1.)

#DNN structure
inputs = Input(shape=(7,))
hidden1 = Dense(512, activation='leaky_relu', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-8), bias_regularizer=regularizers.l2(1e-8), activity_regularizer=regularizers.l2(1e-8))(inputs)
drop1=Dropout(0.2)(hidden1)
hidden2 = Dense(128, activation='leaky_relu', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-8), bias_regularizer=regularizers.l2(1e-8), activity_regularizer=regularizers.l2(1e-8))(drop1)
drop2=Dropout(0.2)(hidden2)
hidden3 = Dense(64, activation='leaky_relu', kernel_regularizer=regularizers.l1_l2(l1=1e-8, l2=1e-8), bias_regularizer=regularizers.l2(1e-8), activity_regularizer=regularizers.l2(1e-8))(drop2)
outputs = Dense(1) (hidden3)

inputs = Input(shape=(7,))
hidden1 = Dense(256, activation='leaky_relu', kernel_regularizer=l2(3e-7), bias_regularizer=l2(3e-7),activity_regularizer=l2(3e-7))(inputs)
drop1=Dropout(0.2)(hidden1)
hidden2 = Dense(128, activation='leaky_relu', kernel_regularizer=l2(3e-7), bias_regularizer=l2(3e-7),activity_regularizer=l2(3e-7))(drop1)
drop2=Dropout(0.2)(hidden2)
hidden3 = Dense(64, activation='leaky_relu', kernel_regularizer=l2(3e-7), bias_regularizer=l2(3e-7),activity_regularizer=l2(3e-7))(drop2)
drop3 = Dropout(0.1)(hidden3)
hidden4 = Dense(32, activation= 'leaky_relu', kernel_regularizer=l2(3e-7), bias_regularizer=l2(3e-7),activity_regularizer=l2(3e-7))(drop3)


outputs = Dense(1) (hidden4)

#model checkpoint and early stopping
filepath = "/home/federico/root/root-6.24.06-install/weights_multiple_eta_model_nuovo.ckpt"
#checkpoint = ModelCheckpoint('saved_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True , mode='min')

callback_list=[checkpoint]

early_stopping = EarlyStopping(monitor = 'val_loss', mode='min' ,patience=200, verbose=2, restore_best_weights = True)
model = Model ( inputs=inputs, outputs=outputs )

#model.add_loss( custom_loss( ,outputs, inputs) )
lr_scheduler = optimizers.schedules.ExponentialDecay(

    initial_learning_rate=8e-7, decay_steps=10000, decay_rate=1.2)
    
#lr_scheduler = optimizers.schedules.PolynomialDecay(0.0001, 10000,0.000000001, power=1.5)

     
opt = optimizers.Adam(learning_rate=lr_scheduler)
#opt = optimizers.SGD(learning_rate=lr_schedule)
model.compile(loss = delta_train, optimizer=opt, metrics=[delta_train ])#,delta_test 'MSE' ])

#write the summary of the network
model.summary()

#plot the network
plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
)

all_inputs_training   = all_inputs_train
all_inputs_validation = all_inputs_validation
transp_training   = transp_train
transp_validation = transp_validation

history = model.fit( all_inputs_training, transp_training, validation_data = (all_inputs_validation,transp_validation),epochs=300, batch_size=128,verbose=2, callbacks=[early_stopping, checkpoint])
np.save('my_history_multiple_eta_model_nuovo.npy',history.history)

#plot the training loss
plt.plot( history.history["loss"], label = 'training loss function' )
#plt.yscale("log")
plt.plot( history.history["val_loss"], label = 'validation loss function')#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test
plt.yscale("log")
#plt.xscale("log")
plt.legend()
plt.title("loss function logaritmic scale")
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.show()

#predicting the target function
transp_predicted_validation_loss = model.predict(all_inputs_validation)
#predicting the train target function after training to use it in the custom loss function
transp_predicted_train_loss = model.predict(all_inputs_train)

#LOADING the model saved with the checpoint method 
#this makes sure that the best weights will be used to make inference
saved_model = Model ( inputs=inputs, outputs=outputs )
saved_model.load_weights(filepath)
saved_model.predict(all_inputs_validation)
#----------------------------------------

#actually  predicting transparency data 
transp_predicted_validation = saved_model.predict(all_inputs_validation)

transp_predicted_train = saved_model.predict(all_inputs_train)
#----------------------------------

plt.plot(transp_predicted_validation, label='predicted', marker = 'p', markersize = 0.5 , linewidth = None)
plt.plot(transp_validation, label='real', marker = 'p',  markersize = 0.5 ,linewidth = None)
plt.legend()

plt.show()
#evaluating metrics and plotting

j=0
new_k = 0
counter_subplot = 1
m=0

fig1 = plt.figure(figsize=(20,20))

for k in fill_sizes:

    Fill_Num = filltest23[m] 
    transp_predicted=[]
    transp_real=[]
    lumi_in_fill_test=[]

    new_k = new_k + k
    

    for i in range (int(j), int(new_k)):

        transp_predicted_prov = transp_predicted_validation[i]
        transp_real_prov = transp_validation[i]
        Lumi_in_fill_prov_test =  infilllumi_test[i]

        transp_predicted = np.append(transp_predicted, transp_predicted_prov)
        transp_real = np.append (transp_real, transp_real_prov)
        lumi_in_fill_test = np.append(lumi_in_fill_test, Lumi_in_fill_prov_test)
        
    j = new_k
    m=m+1
    #calculation of MSE for each fill of validation

    #calculation of MAE for each fill of validation (that gives a more strainghtforward idea of the error for each )
    
    #Plotting transparency for different fills - predicted vs real 

    plt.subplot(3,3,counter_subplot)
    plt.plot(lumi_in_fill_test, transp_predicted, "r--", markersize=2, linewidth=0.75, label="predicted", marker='p')
    plt.plot(lumi_in_fill_test, transp_real, "b--", markersize=2, linewidth=0.75, label="measured", marker='p')
    plt.xlabel("integrated luminosity $Wb^{-1}$ ")
    plt.ylabel("mean transparency")
    plt.tick_params(labelsize=7)
    plt.title(f"fill {Fill_Num}")
    plt.legend()

    counter_subplot=counter_subplot+1
plt.subplots_adjust(left=0.12, bottom=0.1, right=0.5, top=0.5, wspace=0.2, hspace=0.2)
plt.show()

j=0
new_k = 0
counter_subplot = 1
m=0

fig1 = plt.figure(figsize=(20,20))
Mae=0
for k in fill_sizes:

    Fill_Num = filltest23[m] 
    transp_predicted=[]
    transp_real=[]
    lumi_in_fill_test=[]

    new_k = new_k + k


    for i in range (int(j), int(new_k)):

        transp_predicted_prov = transp_predicted_validation[i]
        transp_real_prov = transp_validation[i]
        Lumi_in_fill_prov_test =  infilllumi_test[i]

        transp_predicted = np.append(transp_predicted, transp_predicted_prov)
        transp_real = np.append (transp_real, transp_real_prov)
        lumi_in_fill_test = np.append(lumi_in_fill_test, Lumi_in_fill_prov_test)
        
    j = new_k
    m=m+1

    counter_subplot=counter_subplot+1

plt.show()


#now let's save transparency predictions and real data and metadata into txt files 
#ready to be used in TurnOnCurve.cxx to actually test the model.

#Transparency_predictions = np.vstack([transp_validation,transp_predicted_validation]).T
#output = np.stack((transp_validation, transp_predicted_validation, all_inputs_validation), axis=-1)

np.savetxt(f"{data_folder}/transp_validation_multiple_eta_nuovo.txt", transp_real,fmt="%s ")
np.savetxt(f"{data_folder}/transp_predicted_validation_multiple_eta_nuovo.txt", transp_predicted,fmt="%s ")

#np.savetxt(f"{data_folder}/Transparency_predicitions.txt", Transparency_predictions, fmt="%s %s %s %s")
np.savetxt(f"{data_folder}/Luminosity_data_validation_multiple_eta_nuovo.txt", all_inputs_validation, fmt="%s")



