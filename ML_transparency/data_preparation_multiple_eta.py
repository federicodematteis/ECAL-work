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
        fill=metadata["fill_num"].unique()
        fill = fill[fill != 0]
        
        #excluding NoNSMooTh fills from the "fill" array
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
    
    
        #MEAN TRANSPARENCY in iRing (The target function) for input training
        transp_fill = []
        for k in fill_num:
            df = metadata_fill[metadata_fill.fill_num == k]
            transp = [mean[i] for i in df.index.values]
            transp = transp/transp[0]
            transp_fill = np.append(transp_fill, transp)
            sample_weighting_dimension = np.append(sample_weighting_dimension,len(transp))

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
    
    print('numero di fill usati per il train')
    print(len(FILL_num))

    for k in range (0, len(Norm_time_in_fill_train)):
        
        sample_weighting_dimension
        if Norm_time_in_fill_train[k] == 1. :
            weight = (0.5/(len(FILL_num)))
            weights_train.append(weight)
        else:
            weight = ((1-0.5)/(len(Norm_time_in_fill_train)-(len(FILL_num))))
            weights_train.append(weight)


    #redo sample weighting 
    weights_train1=[]
    for k in range (0, len(Norm_time_in_fill_train)):
        
        sample_weighting_dimension
        if Norm_time_in_fill_train[k] == 1.0 :
            weight = 100
            weights_train1.append(weight)
        else:
            weight = 1
            weights_train1.append(weight)


    print('sample weights control == 1 ?')
    weights_train_def=[]
    norm = sum(weights_train1)
    
    for j in range (0,len(weights_train1)):
        weights_train_def = np.append(weights_train_def, weights_train1[j]/norm)
    print(sum(weights_train_def))
    
        
    #merge metadata for each xtal into a single object
    all_inputs_train = np.stack((instLumi,intLumiLHC,infillLumi,lastfillLumi,filltime,lastpointLumi, ring_index), axis=-1)

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

        #selecting metadata for validation fills 
        metadata_test = metadata[metadata.fill_num.isin(m_value[1])]
        #metadata_test = metadata_test[(metadata_test.lumi_inst >= 0.0001*1e9) & (metadata_test.lumi_inst <= 0.0004*1e9) & (metadata_test.lumi_in_fill >= 0.1*1e9)]
        transp_test = mean[metadata_test.index.values[0]:metadata_test.index.values[0]+len(metadata_test.axes[0])]
        fill_num_test = metadata_test.fill_num.unique()
        print(fill_num_test)
        FILL_num = np.append(FILL_num, fill_num_test)
        #METADATA_test = pd.concat(METADATA_test, metadata_test)
        #METADATA_test.append(metadata_test) 

        #normalizing time_in_fill for test fills
        for k in fill_num_test:
            dftest = metadata_test[metadata_test.fill_num == k]
            Norm_time_in_fill_prov_test = dftest.loc[:,'time_in_fill']
            c=dftest['time_in_fill'].iloc[0]
            Norm_time_in_fill_prov_test = Norm_time_in_fill_prov_test/c
            Norm_time_in_fill_test=np.append(Norm_time_in_fill_test,Norm_time_in_fill_prov_test)
        
        #normalizing transparency
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

    #validation weights for sample weighting 
    #this is a bad sampling 
    for k in range (0, len(Norm_time_in_fill_test)):
        if Norm_time_in_fill_test[k] == 1. :
            weight = 0.5/(len(FILL_num))
            weights_test.append(weight)
        else:
            #weight = (1-0.5)/ (len(fill_num2)+len(merged_norm_time_in_fill)-(len(fill_num)+len(fill_num1)))
            weight = (1-0.5)/(len(Norm_time_in_fill_test)-(len(FILL_num)))
            weights_test.append(weight)
        
        #merge everything into single objects
    all_inputs_test = np.stack((instLumi_test, intLumiLHC_test, infillLumi_test, lastpointLumi_test, filltime_test, lastfillLumi_test, ring_index_test), axis=-1)
    #k_fold_1 = np.stack((), axis=-1)
    #k_fold_2 = np.stack((), axis=-1)

    return [all_inputs_test, transp_test_final, weights_test, metadata_test, true_time_test, Norm_time_in_fill_test, fill_sizes, infillLumi_test]

#function that creates new validation data for cross validation
#def K_fold_cross_validation (dictionary_test, metadata): 

#---------------------------------------------"SMOOTHING"-------------------------------------------------------
#this for is used to hide this section

#--------Fill excluded from iring 22
nonsmooth22 = [ #fill escludi dal train
             5697, 5698, 5699, 5710, 5719, 5733, 5736, 5737, 5738,5739, 5740, 5748,5749, 
             5833,  5834, 5837,       5859, 5860,
             5861, 5862, 5870, 5871, 5873, 5887, 5919, 5920, 5929, 5930, 5946, 5952, 5966, 5970,
             5971, 5974, 6001, 6005, 6006, 6015, 6018, 6021, 6034, 6039, 6044, 6047, 
             6055, 6072, 6130, 6132, 6146, 6155, 6164, 6173, 6179, 6180, 6183,
             6184, 6195, 6199, 6200, 6201, 6217, 6226, 6227, 6228, 6230, 6331, 6232, 6233, 6238,
             6293, 6309, 6313, 6336, 6341, 6351, 6355,
             6374, 6382, 6387, 6388, 6390, 6402, 6431, 6432,
             
            #5704, 5722, 5822, 5824, 5825, 5830, 5838,5874, 5885, 5962, 5963, 5965,
            #5984, 5985, 6041, 6050, 6093, 6094, 6105, 6185,6231, 6236,
            #6141, 6232,
            #6261, 6263, 6269, 6279, 6294, 6377, 6380, 6381, 6386, 6404, 6405,

            #test fills
            5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
            6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143


             ]

#--------Fill excluded from iring 23


nonsmooth23=[ #fill escludi dal train
             5697, 5698, 5699, 5710, 5719, 5733, 5736, 5737, 5738,5739, 5740, 5748,5749, 
             5833,  5834, 5837,       5859, 5860,
             5861, 5862, 5870, 5871, 5873, 5887, 5919, 5920, 5929, 5930, 5946, 5952, 5966, 5970,
             5971, 5974, 6001, 6005, 6006, 6015, 6018, 6021, 6034, 6039, 6044, 6047, 
             6055, 6072, 6130, 6132, 6146, 6155, 6164, 6173, 6179, 6180, 6183,
             6184, 6195, 6199, 6200, 6201, 6217, 6226, 6227, 6228, 6230, 6331, 6232, 6233, 6238,
             6293, 6309, 6313, 6336, 6341, 6351, 6355,
             6374, 6382, 6387, 6388, 6390, 6402, 6431, 6432,
             
            #5704, 5722, 5822, 5824, 5825, 5830, 5838,5874, 5885, 5962, 5963, 5965,
            #5984, 5985, 6041, 6050, 6093, 6094, 6105, 6185,6231, 6236,
            #6141, 6232,
            #6261, 6263, 6269, 6279, 6294, 6377, 6380, 6381, 6386, 6404, 6405,

            #test fills
            5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
            6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143


             ]
             

#--------Fill excluded from iring 24 

                
#new filtering for larger  lumi dataset 
nonsmooth24=[ #fill escludi dal train
             5697, 5698, 5699, 5710, 5719, 5733, 5736, 5737, 5738,5739, 5740, 5748,5749, 
             5833,  5834, 5837,       5859, 5860,
             5861, 5862, 5870, 5871, 5873, 5887, 5919, 5920, 5929, 5930, 5946, 5952, 5966, 5970,
             5971, 5974, 6001, 6005, 6006, 6015, 6018, 6021, 6034, 6039, 6044, 6047, 
             6055, 6072, 6130, 6132, 6146, 6155, 6164, 6173, 6179, 6180, 6183,
             6184, 6195, 6199, 6200, 6201, 6217, 6226, 6227, 6228, 6230, 6331, 6232, 6233, 6238,
             6293, 6309, 6313, 6336, 6341, 6351, 6355,
             6374, 6382, 6387, 6388, 6390, 6402, 6431, 6432,
             
            #5704, 5722, 5822, 5824, 5825, 5830, 5838,5874, 5885, 5962, 5963, 5965,
            #5984, 5985, 6041, 6050, 6093, 6094, 6105, 6185,6231, 6236,
            #6141, 6232,
            #6261, 6263, 6269, 6279, 6294, 6377, 6380, 6381, 6386, 6404, 6405,

            #test fills
            5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
            6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143


             ]

#--------Fill excluded from iring 25
nonsmooth25 = [ #fill escludi dal train
             5697, 5698, 5699, 5710, 5719, 5733, 5736, 5737, 5738,5739, 5740, 5748,5749, 
             5833,  5834, 5837,       5859, 5860,
             5861, 5862, 5870, 5871, 5873, 5887, 5919, 5920, 5929, 5930, 5946, 5952, 5966, 5970,
             5971, 5974, 6001, 6005, 6006, 6015, 6018, 6021, 6034, 6039, 6044, 6047, 
             6055, 6072, 6130, 6132, 6146, 6155, 6164, 6173, 6179, 6180, 6183,
             6184, 6195, 6199, 6200, 6201, 6217, 6226, 6227, 6228, 6230, 6331, 6232, 6233, 6238,
             6293, 6309, 6313, 6336, 6341, 6351, 6355,
             6374, 6382, 6387, 6388, 6390, 6402, 6431, 6432,
             
            #5704, 5722, 5822, 5824, 5825, 5830, 5838,5874, 5885, 5962, 5963, 5965,
            #5984, 5985, 6041, 6050, 6093, 6094, 6105, 6185,6231, 6236,
            #6141, 6232,
            #6261, 6263, 6269, 6279, 6294, 6377, 6380, 6381, 6386, 6404, 6405,

            #test fills
            5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
            6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143


             ]           
    #with this command we actually exclude nonsmooth fills from the "fill" array 
    #but we are also excluding validation fills which we insert in the validation dictionary (dictionary_test).
#--------Fill excluded from iring 26
nonsmooth26 = [  #fill escludi dal train
             5697, 5698, 5699, 5710, 5719, 5733, 5736, 5737, 5738,5739, 5740, 5748,5749, 
             5833,  5834, 5837,       5859, 5860,
             5861, 5862, 5870, 5871, 5873, 5887, 5919, 5920, 5929, 5930, 5946, 5952, 5966, 5970,
             5971, 5974, 6001, 6005, 6006, 6015, 6018, 6021, 6034, 6039, 6044, 6047, 
             6055, 6072, 6130, 6132, 6146, 6155, 6164, 6173, 6179, 6180, 6183,
             6184, 6195, 6199, 6200, 6201, 6217, 6226, 6227, 6228, 6230, 6331, 6232, 6233, 6238,
             6293, 6309, 6313, 6336, 6341, 6351, 6355,
             6374, 6382, 6387, 6388, 6390, 6402, 6431, 6432,
             
            #5704, 5722, 5822, 5824, 5825, 5830, 5838,5874, 5885, 5962, 5963, 5965,
            #5984, 5985, 6041, 6050, 6093, 6094, 6105, 6185,6231, 6236,
            #6141, 6232,
            #6261, 6263, 6269, 6279, 6294, 6377, 6380, 6381, 6386, 6404, 6405,

            #test fills
            5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
            6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143


             ]
                

#--------Fill excluded from iring 27
nonsmooth27 = [ #fill escludi dal train
             5697, 5698, 5699, 5710, 5719, 5733, 5736, 5737, 5738,5739, 5740, 5748,5749, 
             5833,  5834, 5837,       5859, 5860,
             5861, 5862, 5870, 5871, 5873, 5887, 5919, 5920, 5929, 5930, 5946, 5952, 5966, 5970,
             5971, 5974, 6001, 6005, 6006, 6015, 6018, 6021, 6034, 6039, 6044, 6047, 
             6055, 6072, 6130, 6132, 6146, 6155, 6164, 6173, 6179, 6180, 6183,
             6184, 6195, 6199, 6200, 6201, 6217, 6226, 6227, 6228, 6230, 6331, 6232, 6233, 6238,
             6293, 6309, 6313, 6336, 6341, 6351, 6355,
             6374, 6382, 6387, 6388, 6390, 6402, 6431, 6432,
             
            #5704, 5722, 5822, 5824, 5825, 5830, 5838,5874, 5885, 5962, 5963, 5965,
            #5984, 5985, 6041, 6050, 6093, 6094, 6105, 6185,6231, 6236,
            #6141, 6232,
            #6261, 6263, 6269, 6279, 6294, 6377, 6380, 6381, 6386, 6404, 6405,

            #test fills
            5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
            6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143


             ]
            

#--------Fill excluded from iring 28
nonsmooth28 = [ #fill escludi dal train
             5697, 5698, 5699, 5710, 5719, 5733, 5736, 5737, 5738,5739, 5740, 5748,5749, 
             5833,  5834, 5837,       5859, 5860,
             5861, 5862, 5870, 5871, 5873, 5887, 5919, 5920, 5929, 5930, 5946, 5952, 5966, 5970,
             5971, 5974, 6001, 6005, 6006, 6015, 6018, 6021, 6034, 6039, 6044, 6047, 
             6055, 6072, 6130, 6132, 6146, 6155, 6164, 6173, 6179, 6180, 6183,
             6184, 6195, 6199, 6200, 6201, 6217, 6226, 6227, 6228, 6230, 6331, 6232, 6233, 6238,
             6293, 6309, 6313, 6336, 6341, 6351, 6355,
             6374, 6382, 6387, 6388, 6390, 6402, 6431, 6432,
             
            #5704, 5722, 5822, 5824, 5825, 5830, 5838,5874, 5885, 5962, 5963, 5965,
            #5984, 5985, 6041, 6050, 6093, 6094, 6105, 6185,6231, 6236,
            #6141, 6232,
            #6261, 6263, 6269, 6279, 6294, 6377, 6380, 6381, 6386, 6404, 6405,

            #test fills
            5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
            6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143


             ]
            


#---------------------

filltest18 = [5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
                6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143]
filltest19 = [5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
                6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143]
filltest20 = [5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
                6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143]
filltest21 = [5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
                6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143]

filltest22 = [5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
                6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143]
filltest23 = [5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
                6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143]
filltest24 = [5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
                6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143]
filltest25 = [5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
                6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143]
filltest26 = [5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
                6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143]
filltest27 = [5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
                6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143]
filltest28 = [5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
                6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143]

nonsmooth18=[ #fill escludi dal train
             5697, 5698, 5699, 5710, 5719, 5733, 5736, 5737, 5738,5739, 5740, 5748,5749, 
             5833,  5834, 5837,       5859, 5860,
             5861, 5862, 5870, 5871, 5873, 5887, 5919, 5920, 5929, 5930, 5946, 5952, 5966, 5970,
             5971, 5974, 6001, 6005, 6006, 6015, 6018, 6021, 6034, 6039, 6044, 6047, 
             6055, 6072, 6130, 6132, 6146, 6155, 6164, 6173, 6179, 6180, 6183,
             6184, 6195, 6199, 6200, 6201, 6217, 6226, 6227, 6228, 6230, 6331, 6232, 6233, 6238,
             6293, 6309, 6313, 6336, 6341, 6351, 6355,
             6374, 6382, 6387, 6388, 6390, 6402, 6431, 6432,
             
            #5704, 5722, 5822, 5824, 5825, 5830, 5838,5874, 5885, 5962, 5963, 5965,
            #5984, 5985, 6041, 6050, 6093, 6094, 6105, 6185,6231, 6236,
            #6141, 6232,
            #6261, 6263, 6269, 6279, 6294, 6377, 6380, 6381, 6386, 6404, 6405,

            #test fills
            5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
            6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143


             ]
nonsmooth19=[ #fill escludi dal train
             5697, 5698, 5699, 5710, 5719, 5733, 5736, 5737, 5738,5739, 5740, 5748,5749, 
             5833,  5834, 5837,       5859, 5860,
             5861, 5862, 5870, 5871, 5873, 5887, 5919, 5920, 5929, 5930, 5946, 5952, 5966, 5970,
             5971, 5974, 6001, 6005, 6006, 6015, 6018, 6021, 6034, 6039, 6044, 6047, 
             6055, 6072, 6130, 6132, 6146, 6155, 6164, 6173, 6179, 6180, 6183,
             6184, 6195, 6199, 6200, 6201, 6217, 6226, 6227, 6228, 6230, 6331, 6232, 6233, 6238,
             6293, 6309, 6313, 6336, 6341, 6351, 6355,
             6374, 6382, 6387, 6388, 6390, 6402, 6431, 6432,
             
            #5704, 5722, 5822, 5824, 5825, 5830, 5838,5874, 5885, 5962, 5963, 5965,
            #5984, 5985, 6041, 6050, 6093, 6094, 6105, 6185,6231, 6236,
            #6141, 6232,
            #6261, 6263, 6269, 6279, 6294, 6377, 6380, 6381, 6386, 6404, 6405,

            #test fills
            5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
            6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143


             ]
nonsmooth20=[ #fill escludi dal train
             5697, 5698, 5699, 5710, 5719, 5733, 5736, 5737, 5738,5739, 5740, 5748,5749, 
             5833,  5834, 5837,       5859, 5860,
             5861, 5862, 5870, 5871, 5873, 5887, 5919, 5920, 5929, 5930, 5946, 5952, 5966, 5970,
             5971, 5974, 6001, 6005, 6006, 6015, 6018, 6021, 6034, 6039, 6044, 6047, 
             6055, 6072, 6130, 6132, 6146, 6155, 6164, 6173, 6179, 6180, 6183,
             6184, 6195, 6199, 6200, 6201, 6217, 6226, 6227, 6228, 6230, 6331, 6232, 6233, 6238,
             6293, 6309, 6313, 6336, 6341, 6351, 6355,
             6374, 6382, 6387, 6388, 6390, 6402, 6431, 6432,
             
            #5704, 5722, 5822, 5824, 5825, 5830, 5838,5874, 5885, 5962, 5963, 5965,
            #5984, 5985, 6041, 6050, 6093, 6094, 6105, 6185,6231, 6236,
            #6141, 6232,
            #6261, 6263, 6269, 6279, 6294, 6377, 6380, 6381, 6386, 6404, 6405,

            #test fills
            5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
            6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143


             ]
nonsmooth21=[ #fill escludi dal train
             5697, 5698, 5699, 5710, 5719, 5733, 5736, 5737, 5738,5739, 5740, 5748,5749, 
             5833,  5834, 5837,       5859, 5860,
             5861, 5862, 5870, 5871, 5873, 5887, 5919, 5920, 5929, 5930, 5946, 5952, 5966, 5970,
             5971, 5974, 6001, 6005, 6006, 6015, 6018, 6021, 6034, 6039, 6044, 6047, 
             6055, 6072, 6130, 6132, 6146, 6155, 6164, 6173, 6179, 6180, 6183,
             6184, 6195, 6199, 6200, 6201, 6217, 6226, 6227, 6228, 6230, 6331, 6232, 6233, 6238,
             6293, 6309, 6313, 6336, 6341, 6351, 6355,
             6374, 6382, 6387, 6388, 6390, 6402, 6431, 6432,
             
            #5704, 5722, 5822, 5824, 5825, 5830, 5838,5874, 5885, 5962, 5963, 5965,
            #5984, 5985, 6041, 6050, 6093, 6094, 6105, 6185,6231, 6236,
            #6141, 6232,
            #6261, 6263, 6269, 6279, 6294, 6377, 6380, 6381, 6386, 6404, 6405,

            #test fills
            5848, 6053 ,6140, 6174, 6191, 6243, 6275,  6300, 6356,5958, 6031, 6046, 6053,6110,6324,
            6356,6371,6298, 6318, 6343, 6346, 6358, 6362,6255, 6291, 6307,6314, 6143


             ]


#k_fold_1 = [6384, 6385, 6386, 6389]
#every fill used for cross validation is to be excluded from training 
#take it simple : in this cross validation : 1 fill = 1 sample
# the number of sample is the number of k-fold

#the best thing to organize the space work might be a file json to control fills for validation 
#and train and npy files for transparency data

#running the functions below we have data ready for the model training
# #for first we instance training and validation dictionaries 
dictionary_train = {18:["iRing18new.npy", nonsmooth18], 19:["iRing19new.npy", nonsmooth19], 20:["iRing20new.npy", nonsmooth20], 21:["iRing21new.npy", nonsmooth21], 22:["iRing22new.npy", nonsmooth22], 23:["iRing23new.npy", nonsmooth23], 24:["iRing24new.npy", nonsmooth24], 25:["iRing25new.npy", nonsmooth25], 26:["iRing26new.npy", nonsmooth26], 27:["iRing27new.npy", nonsmooth27], 28:["iRing28new.npy", nonsmooth28]}
#dictionary_train = { 22:["iRing22new.npy", nonsmooth22], 25:["iRing25new.npy", nonsmooth25], 28:["iRing28new.npy", nonsmooth28]}
#dictionary_train = { 23:["iRing23new.npy", nonsmooth23]}

dictionary_test = {18:["iRing18new.npy", filltest18],19:["iRing19new.npy", filltest19], 20:["iRing20new.npy", filltest20], 21:["iRing21new.npy", filltest21],22:["iRing22new.npy", filltest22], 23:["iRing23new.npy", filltest23], 24:["iRing24new.npy", filltest24], 25:["iRing25new.npy", filltest25], 26:["iRing26new.npy", filltest26], 27:["iRing27new.npy", filltest27], 28:["iRing28new.npy", filltest28]}
#dictionary_cross_validation ={23:["iRing23new.npy", k_fold_1] }
#we instance a second vector to be used for cross valiating the model
metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")


#Pre-processing data for training and validation
[all_inputs_train, transp_train, weights_train, norm_time_in_fill_train]=Pre_Processing_Train(dictionary_train, metadata)
[all_inputs_validation, transp_validation, weights_validation, metadata_test, time_test, norm_time_in_fill_test, fill_sizes, infilllumi_test]=Pre_Processing_Test(dictionary_test,metadata)
