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


#---------------------------------------------"SMOOTHING"-------------------------------------------------------

#in questa sezione seleziono per ogni i-Ring gli LHC fill che voglio escludere dall'analisi

#--------Fill excluded from iring 24 
nonsmooth24 = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5883, 5887, 5954,
                5984, 6030, 6041, 6057, 6084, 6089, 6090, 6091, 6096, 6105, 6106, 6116, 6152, 
                6159, 6160, 6167, 6168, 6170, 6189, 6192, 6193, 6261, 6262, 6263, 6279, 
                6318,
                 #we exclude some fills for validation for iring 24
                6356, 6358, 6360, 6362, 6364, 6370, 6371
                ]
                
#new filtering for larger  lumi dataset 
nonsmooth24=[5697, 5698, 5699, 5710, 5719, 5736, 5739, 5740, 5749, 5859, 5860,
             5861, 5862, 5871, 5887, 5919, 5929, 5930, 5946, 5952, 5966, 5970,
             5971, 5974, 6001, 6005, 6006, 6015, 6018, 6021, 6034, 6039, 6047, 
             6055, 6072, 6130, 6132, 6146, 6155, 6164, 6173, 6179, 6180, 6183,
             6184, 6200, 6201, 6217, 6233, 6293, 6309, 6313, 6341, 6351, 6355,
             6374, 6382, 6387, 6388, 6390, 6402, 6431, 6432,
             
             6371
             #6272,
             #6295
#function that creates dataset ready for the training 
def Pre_Processing_Train(dictionary_train,metadata):

    transp_train = []
    instLumi = []
    intLumiLHC = []
    infillLumi = []
    lastfillLumi = []
    lastpointLumi = []
    filltime = []
    true_time = []
    ring_index = []

    Norm_time_in_fill_train = []
    Norm_time_in_fill_prov_train = []
    weights_train = []
    FILL_num = []
    METADATA_train=[]

    sample_weighting_dimension =[]

    for i_key, i_value in dictionary_train.items():
        print(i_value[0])

        data=np.load(f"{data_folder}/{i_value[0]}")
        data_df = pd.DataFrame(data)
        data_df.head()
    
        mean=[]
        for k in range (0, len(data_df.axes[1])):
            mean = np.append(mean, np.mean(data_df[k]))

        #mean transparency in iring
        mean=mean[mean != -1]
        metadata = metadata.iloc[:len(mean)][mean != -1]

        #selecting metadata for fill (locking metadata to in_fill=1)
        #fill arrays are lists of fill_num related to each crystal
        fill=metadata["fill_num"].unique()
        fill = fill[fill != 0]
        
        #excluding NoNSMooTh fills from the "fill" array
        #non va bene definire così i fill da escludere
        #devo far corrispondere a value di NoNSMooth value di training data
        NONsMOOTH=i_value[1]
        for iev in range (0, len(NONsMOOTH)) :
            fill = fill[fill != NONsMOOTH[iev]]

        metadata_fill = metadata[metadata.fill_num.isin(fill)]
        #metadata_fill = metadata_fill[(metadata_fill.lumi_inst >= 0.0001*1e9) & (metadata_fill.lumi_inst <= 0.0004*1e9) & (metadata_fill.lumi_in_fill >= 0.1*1e9)]
        fill_num = metadata_fill.fill_num.unique()
        FILL_num = np.append(FILL_num, fill_num)
        METADATA_train=np.append(METADATA_train, metadata_fill)

        #Normalised time in fill for xtal 23 (iring23)
        for k in fill_num:
            dftrain = metadata_fill[metadata_fill.fill_num == k]
            #time_in_fill normalizzato per ogni fill, rispetto all'istante iniziale
            Norm_time_in_fill_prov_train = dftrain.loc[:,'time_in_fill']
            Norm_time_in_fill_prov_train = Norm_time_in_fill_prov_train/dftrain['time_in_fill'].iloc[0]
            Norm_time_in_fill_train=np.append(Norm_time_in_fill_train,Norm_time_in_fill_prov_train)
    
    
        #MEAN TRANSPARENCY in iRing (The target function)
        
        transp_fill = []
        for k in fill_num:
            df = metadata_fill[metadata_fill.fill_num == k]
            transp = [mean[i] for i in df.index.values]
            transp = transp/transp[0]
            transp_fill = np.append(transp_fill, transp)
            sample_weighting_dimension = np.append(sample_weighting_dimension,len(transp))

        print(len(sample_weighting_dimension))
        print(sample_weighting_dimension)
        transp_train = np.append(transp_train,transp_fill)
        #--------------------------------------TRAINING DATA------------------------------------------------------------
        #Metadata (input) for training related to each xtal
        #fill_num is different for subdataset related to each crystal,
        #it depends on which fills we excluded form the train

        instLumi_p = (1e-9)*metadata_fill.loc[:,'lumi_inst']
        intLumiLHC_p = (1e-9)*metadata_fill.loc[:,'lumi_int']
        infillLumi_p = (1e-9)*metadata_fill.loc[:,'lumi_in_fill']
        lastfillLumi_p = (1e-9)*metadata_fill.loc[:,'lumi_last_fill']
        filltime_p = (1e-9)*metadata_fill.loc[:,'time_in_fill']
        lastpointLumi_p = (1e-9)*metadata_fill.loc[:, 'lumi_since_last_point']
        true_time_p = (1e-9)*metadata_fill.loc[:, 'time']
        ring_index_p = np.zeros(len(metadata_fill))
        print(i_key)
        for j in range (0, len(metadata_fill)):
            if i_key ==18:
                ring_index_p[j] = 1.981
            if i_key ==19:
                ring_index_p[j] = 2.014
            if i_key ==20:
                ring_index_p[j] = 2.048
            if i_key ==21:
                ring_index_p[j] = 2.082
            if i_key ==22:
                ring_index_p[j] = 2.119
            if i_key ==23:
                ring_index_p[j] = 2.157
            if i_key ==24:
                ring_index_p[j] = 2.196
            if i_key ==25:
                ring_index_p[j] = 2.236
            if i_key ==26:
                ring_index_p[j] = 2.278
            if i_key == 27:
                ring_index_p[j] = 2.322
            if i_key == 28:
                ring_index_p[j] = 2.368
            

        
        
        #merge data into a single object
        instLumi=np.append(instLumi, instLumi_p)
        intLumiLHC=np.append(intLumiLHC, intLumiLHC_p)
        infillLumi=np.append(infillLumi, infillLumi_p)
        lastfillLumi=np.append(lastfillLumi, lastfillLumi_p)
        filltime=np.append(filltime, filltime_p)
        lastpointLumi=np.append(lastpointLumi, lastpointLumi_p)
        true_time=np.append(true_time, true_time_p)
        ring_index = np.append(ring_index, ring_index_p)



    #Virtual weights for samplle weighting 
    #len(FILL_num) is the number of first instances over all fills used for training, for various xtals.
    #len(Norm_time_in_fill) is the total number of instances for all fills, for various xtals.
    print(len(FILL_num))
    #this is a bad sample weighting 
    #it weights transparency values of 1 but is not true that only the first instance in the fill has T=1

    for k in range (0, len(Norm_time_in_fill_train)):
        
        sample_weighting_dimension
        if Norm_time_in_fill_train[k] == 1. :
            weight = (0.5/(len(FILL_num)))
            weights_train.append(weight)
        else:
            #weight = (1-0.5)/ (len(fill_num2)+len(merged_norm_time_in_fill)-(len(fill_num)+len(fill_num1)))
            weight = ((1-0.5)/(len(Norm_time_in_fill_train)-(len(FILL_num))))
            weights_train.append(weight)


    
    #prec_dimension=0

    # for dimension in sample_weighting_dimension:
    #         #ho la dim del fill i esimo
    #     for k in range(prec_dimension,Norm_time_in_fill_train[len(dimension)]) :
        
    #         while k < len(dimension):
    #             dF
    #     prec_dimension = dimension+1
    #     #for k in range (0,dimension):

    #redo sample weighting 
    weights_train1=[]
    for k in range (0, len(Norm_time_in_fill_train)):
        
        sample_weighting_dimension
        if Norm_time_in_fill_train[k] == 1.0 :
            weight = 100
           # (len(FILL_num)))
            weights_train1.append(weight)
            
        else:
            #weight = (1-0.5)/ (len(fill_num2)+len(merged_norm_time_in_fill)-(len(fill_num)+len(fill_num1)))
            weight = 1
            #(len(Norm_time_in_fill_train)-(len(FILL_num))))
            weights_train1.append(weight)


    print('weightscontrol')
    weights_train_def=[]
    norm = sum(weights_train1)
    #print(0.1*(len(Norm_time_in_fill_train)-(len(FILL_num)))+1*(len(FILL_num)))
    
    for j in range (0,len(weights_train1)):
        weights_train_def = np.append(weights_train_def, weights_train1[j]/norm)
    print(sum(weights_train_def))
    
        
    #merge metadata for each xtal into a single object
    all_inputs_train = np.stack((instLumi,intLumiLHC,infillLumi,lastfillLumi,filltime,lastpointLumi), axis=-1)

    return [all_inputs_train, transp_train, weights_train_def, Norm_time_in_fill_train]

#function that creates dataset ready for inference
def Pre_Processing_Test(dictionary_test,metadata):
    transp_test_final = []

    instLumi_test = []
    intLumiLHC_test = []
    infillLumi_test = []
    lastfillLumi_test = []
    filltime_test = []
    lastpointLumi_test = []
    true_time_test = []
    ring_index_test = []

    Norm_time_in_fill_test = []
    Norm_time_in_fill_prov_test = []
    weights_test = []
    FILL_num = []
    METADATA_test=[]
    fill_sizes=[]
    
    #scorro il dizionario con un for, a ogni iterazione seleziono un i-Ring differente nelle Endcaps
    for m_key, m_value in dictionary_test.items():

        data=np.load(f"{data_folder}/{m_value[0]}")
        data_df = pd.DataFrame(data)
        data_df.head()

        mean=[]

        for k in range (0, len(data_df.axes[1])):
            mean = np.append(mean, np.mean(data_df[k]))

        #mean transparency in iring selected for validation
        mean=mean[mean != -1]
        metadata = metadata.iloc[:len(mean)][mean != -1]

        #selecting luminosity metadata for validation fills 
        metadata_test = metadata[metadata.fill_num.isin(m_value[1])]
        #metadata_test = metadata_test[(metadata_test.lumi_inst >= 0.0001*1e9) & (metadata_test.lumi_inst <= 0.0004*1e9) & (metadata_test.lumi_in_fill >= 0.1*1e9)]
        transp_test = mean[metadata_test.index.values[0]:metadata_test.index.values[0]+len(metadata_test.axes[0])]
        fill_num_test = metadata_test.fill_num.unique()
        print(fill_num_test)
        FILL_num = np.append(FILL_num, fill_num_test)

        #normalizing time_in_fill for test fills
        #time in fill ci serve per selezionare la prima istanza in ogni fill
        for k in fill_num_test:
            dftest = metadata_test[metadata_test.fill_num == k]
            Norm_time_in_fill_prov_test = dftest.loc[:,'time_in_fill']
            c=dftest['time_in_fill'].iloc[0]
            Norm_time_in_fill_prov_test = Norm_time_in_fill_prov_test/c
            Norm_time_in_fill_test=np.append(Norm_time_in_fill_test,Norm_time_in_fill_prov_test)
        
        
        #normalizing transparency at the beginning of each LHC fill
        for k in fill_num_test:
            fill_sizes_prov=[]
            df_test = metadata_test[metadata_test.fill_num == k]
            transp_test = [mean[i] for i in df_test.index.values]
            transp_test = transp_test/transp_test[0]
            #size of each fill used for testing 
            fill_sizes_prov = len(transp_test)
            fill_sizes = np.append(fill_sizes, fill_sizes_prov)
            transp_test_final = np.append(transp_test_final, transp_test)

        instLumi_test_p = (1e-9)*metadata_test.loc[:,'lumi_inst']
        intLumiLHC_test_p = (1e-9)*metadata_test.loc[:,'lumi_int']
        infillLumi_test_p = (1e-9)*metadata_test.loc[:,'lumi_in_fill']
        lastfillLumi_test_P = (1e-9)*metadata_test.loc[:,'lumi_last_fill']
        filltime_test_p = (1e-9)*metadata_test.loc[:,'time_in_fill']
        lastpointLumi_test_p = (1e-9)*metadata_test.loc[:, 'lumi_since_last_point']
        true_time_test_p = (1e-9)*metadata_test.loc[:, 'time']
        ring_index_test_p = np.zeros(len(metadata_test))
        #per ogni i-Ring associo la propria pseudorapidità 
        for j in range (0, len(metadata_test)):
            if m_key ==18:
                ring_index_test_p[j] = 1.981
            if m_key ==19:
                ring_index_test_p[j] = 2.014
            if m_key ==20:
                ring_index_test_p[j] = 2.048
            if m_key ==21:
                ring_index_test_p[j] = 2.082
            if m_key ==22:
                ring_index_test_p[j] = 2.119
            if m_key ==23:
                ring_index_test_p[j] = 2.157
            if m_key ==24:
                ring_index_test_p[j] = 2.196
            if m_key ==25:
                ring_index_test_p[j] = 2.236
            if m_key ==26:
                ring_index_test_p[j] = 2.278
            if m_key == 27:
                ring_index_test_p[j] = 2.322
            if m_key == 28:
                ring_index_test_p[j] = 2.368
            

        instLumi_test=np.append(instLumi_test, instLumi_test_p)
        intLumiLHC_test=np.append(intLumiLHC_test,intLumiLHC_test_p)
        infillLumi_test=np.append(infillLumi_test,infillLumi_test_p)
        lastfillLumi_test=np.append(lastfillLumi_test,lastfillLumi_test_P)
        filltime_test=np.append(filltime_test,filltime_test_p)
        lastpointLumi_test=np.append(lastpointLumi_test,lastpointLumi_test_p)
        true_time_test=np.append(true_time_test,true_time_test_p)
        ring_index_test=np.append(ring_index_test,ring_index_test_p)

    
        
    #features da dare alla rete neurale in un singolo oggetto che passeremo al layer di input
    all_inputs_test = np.stack((instLumi_test, intLumiLHC_test, infillLumi_test,lastfillLumi_test , filltime_test, lastpointLumi_test), axis=-1)
    #k_fold_1 = np.stack((), axis=-1)
    #k_fold_2 = np.stack((), axis=-1)

    return [all_inputs_test, transp_test_final, weights_test, metadata_test, true_time_test, Norm_time_in_fill_test, fill_sizes, infillLumi_test]

 
#--------------------------------Machine Learning ─=≡Σ(([ ⊐•̀⌂•́]⊐-----------------------------------------------

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
def clip(x, max_value=1.2):
    return activations.relu(x, max_value=1.2)


#struttura della rete neurale usata

#DNN structure

inputs = Input(shape=(6,))
hidden1 = Dense(256, activation='leaky_relu', kernel_regularizer=l2(3e-8), bias_regularizer=regularizers.l2(3e-8),activity_regularizer=l2(3e-8))(inputs)
drop1=Dropout(0.2)(hidden1)
hidden2 = Dense(128, activation='leaky_relu', kernel_regularizer=l2(3e-8), bias_regularizer=regularizers.l2(3e-8),activity_regularizer=l2(3e-8))(drop1)
drop2=Dropout(0.2)(hidden2)
hidden3 = Dense(64, activation='leaky_relu', kernel_regularizer=l2(3e-8), bias_regularizer=regularizers.l2(3e-8),activity_regularizer=l2(3e-8))(drop2)
outputs = Dense(1) (hidden3)

#model checkpoint and early stopping
filepath = "/home/federico/root/root-6.24.06-install/weights_single_eta_model_L2e-8.ckpt"
#checkpoint = ModelCheckpoint('saved_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True , mode='min')

callback_list=[checkpoint]

#monitor della loss di validation per evitare overfitting su dataset di train 
early_stopping = EarlyStopping(monitor = 'val_loss', mode='min' ,patience=300, verbose=2, restore_best_weights = True)
model = Model ( inputs=inputs, outputs=outputs )

#model.add_loss( custom_loss( ,outputs, inputs) )
lr_scheduler = optimizers.schedules.ExponentialDecay(

    initial_learning_rate=3e-6, decay_steps=10000, decay_rate=1.2)
     
#l'algoritmo di otimizzazione usato è un ADAM optimizer con uno scheduler per controllare il decadimento del learning rate 
opt = optimizers.Adam(learning_rate=lr_scheduler)
#opt = optimizers.SGD(learning_rate=lr_scheduler)
model.compile(loss = delta_train, optimizer=opt, metrics=[ delta_train ])

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

#training della rete
history = model.fit( all_inputs_training, transp_training, validation_data = (all_inputs_validation,transp_validation),epochs=900, batch_size=32,verbose=2, callbacks=[early_stopping, checkpoint])
np.save('my_history_single_eta_model_L2e-8.npy',history.history)

#plot the training loss
plt.plot( history.history["loss"], label = 'training loss function' )
#plt.yscale("log")
plt.plot( history.history["val_loss"], label = 'validation loss function' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test
plt.yscale("log")
plt.legend()
plt.title("loss function logaritmic scale")
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.show()

#plotting the learning rate 
# lrs = [lr_scheduler(step) for step in range(10000)]

# plt.scatter(range(10000), lrs, "b--", markersize=2, linewidth=0.75, marker='p')
# plt.title("ExponentialDecay")
# plt.xlabel("Steps")
# plt.ylabel("Learning Rate")
# plt.show()

#-------------------------------Inference phase-------------------------------
#questa prima fase di inferenza ci permette di ottenere le predizioni a ogni epoca 
#per usarle nella custom loss function

#predicting the target function
transp_predicted_validation_loss = model.predict(all_inputs_validation)
#predicting the train target function after training to use it in the custom loss function
transp_predicted_train_loss = model.predict(all_inputs_train)

#----------------------------------------
#In questa seconda fase di inferenza andiamo a effettuare le predizioni usando
#i pesi del modello con la migliore configurazione trovata durante la back-propagation

#LOADING the model saved with the checpoint method 
#this makes sure that the best weights will be used to make inference
saved_model = Model ( inputs=inputs, outputs=outputs )
saved_model.load_weights(filepath)
saved_model.predict(all_inputs_validation)


#actually  predicting transparency data 
transp_predicted_validation = saved_model.predict(all_inputs_validation)

transp_predicted_train = saved_model.predict(all_inputs_train)
#----------------------------------

#alcuni script di plotting usando i dizionari 
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
    #calculation of MSE for each fill of validation

    #calculation of MAE for each fill of validation (that gives a more strainghtforward idea of the error for each ) 
#    Mae = (Mae + abs(transp_real-transp_predicted)/transp_real)/ 2
    #plt.subplot(3,3,counter_subplot)
    # plt.plot(lumi_in_fill_test, Mae, markersize=2, linewidth=0.75, label=(f'{Fill_Num}'), marker='p')
    # plt.xlabel("integrated luminosity $Wb^{-1}$ ")
    # plt.ylabel("relative absolute error")
    # plt.tick_params(labelsize=7)
    # #plt.title(f"fill {Fill_Num}")
    # plt.legend()
    
    counter_subplot=counter_subplot+1

plt.show()


#now let's save transparency predictions and real data and metadata into txt files 
#ready to be used in TurnOnCurve.cxx to actually test the model.

#Transparency_predictions = np.vstack([transp_validation,transp_predicted_validation]).T
#output = np.stack((transp_validation, transp_predicted_validation, all_inputs_validation), axis=-1)

np.savetxt(f"{data_folder}/transp_validation_single_eta_model_L2e-6.txt", transp_real,fmt="%s ")
np.savetxt(f"{data_folder}/transp_predicted_single_eta_model_L2e-6.txt", transp_predicted,fmt="%s ")

#np.savetxt(f"{data_folder}/Transparency_predicitions.txt", Transparency_predictions, fmt="%s %s %s %s")
np.savetxt(f"{data_folder}/Luminosity_data_validation_single_eta_model_L2e-6.txt", all_inputs_validation, fmt="%s")

#Improtanza delle features udate in input al modello per il training 
#interpretability tools for machine learning 

import lime
import lime.lime_tabular
from sklearn import linear_model

#set up the LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(all_inputs_validation,
                                                  training_labels = None,
                                                  feature_names = ['lumi_inst', 'lumi_in_fill', 'lumi_LHC', 'time_in_fill', 'lumi_last_fill', 'lumi_last_point'],
                                                  #feature_selection='lasso_path',
                                                  mode = 'regression',
                                                  discretize_continuous = False)

# you need to modify the output since keras outputs a tensor and LIME takes arrays
def predict(x):
    return model.predict(x).flatten()

# compute the explainer. Chose Huber for its robustness against outliers
i=1
exp = explainer.explain_instance(all_inputs_train[i,:],
                                  predict,
                                  num_features=6,
                                  distance_metric='euclidean',
                                 num_samples=len(all_inputs_train),
                                 model_regressor = linear_model.HuberRegressor())

# generate plot for one item
exp.show_in_notebook(show_table=True, predict_proba=True, show_predicted_value=True)
[exp.as_pyplot_figure(label=1)]
#plt.figure()
plt.show()

#SP LIME 
from lime import submodular_pick
#set up sp lime with 20 samples. The more amount of samples time increases dramatically
sp_obj = submodular_pick.SubmodularPick(explainer, 
                                        all_inputs_validation,
                                        predict, 
                                        sample_size=len(all_inputs_validation),
                                        num_features=6,
                                        num_exps_desired=5)

#get explanation matrix
W_matrix = pd.DataFrame([dict(this.as_list()) for this in sp_obj.explanations])

#get overall mean explanation for each feature
matrix_mean = W_matrix.mean()
plt.figure(figsize=(6,6))

matrix_mean.sort_values(ascending=False).plot.bar()
plt.show()

