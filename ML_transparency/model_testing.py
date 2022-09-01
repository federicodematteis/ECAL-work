#automate data preparation with few functions 
#baseline setup
from ast import In
from cmath import exp
from itertools import tee
from random import sample
from sqlite3 import Time
from tabnanny import check
from tkinter.ttk import LabelFrame
from importlib_metadata import Sectioned
from matplotlib import artist
#from matplotlib.lines import _LineStyle
import matplotlib.pyplot as plt
import ROOT
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
import seaborn as sns
from matplotlib.patches import Rectangle

from keras.regularizers import l2
from keras.regularizers import l1
from tensorflow.keras.models import load_model
import matplotlib.patches as mpatches
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

    Date=[]

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

        #date of events
        #date = [datetime.datetime.fromtimestamp(ts) for ts in metadata.time]

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
        date_fill = [datetime.datetime.fromtimestamp(ts) for ts in metadata_fill.time]
        #metadata_fill = metadata_fill[(metadata_fill.lumi_inst >= 0.0001*1e9) & (metadata_fill.lumi_inst <= 0.0004*1e9) & (metadata_fill.lumi_in_fill >= 0.1*1e9)]
        fill_num = metadata_fill.fill_num.unique()
        FILL_num = np.append(FILL_num, fill_num)
        #METADATA_train=np.append(METADATA_train, metadata_fill)
      
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
        #date = date[metadata_fill.index.values[0]:metadata_fill.index.values[0]+len(metadata_fill.axes[0])]
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
                ring_index_p[j] = 1.479
            if i_key ==19:
                ring_index_p[j] = 1.566
            if i_key ==20:
                ring_index_p[j] = 1.653
            if i_key ==21:
                ring_index_p[j] = 1.740
            if i_key ==22:
                ring_index_p[j] = 1.830
            if i_key ==23:
                ring_index_p[j] = 1.930
            if i_key ==24:
                ring_index_p[j] = 2.043
            if i_key ==25:
                ring_index_p[j] = 2.172
            if i_key ==26:
                ring_index_p[j] = 2.322
            if i_key == 27:
                ring_index_p[j] = 2.500
            if i_key == 28:
                ring_index_p[j] = 2.650
        
        #merge data into a single object
        instLumi=np.append(instLumi, instLumi_p)
        intLumiLHC=np.append(intLumiLHC, intLumiLHC_p)
        infillLumi=np.append(infillLumi, infillLumi_p)
        lastfillLumi=np.append(lastfillLumi, lastfillLumi_p)
        filltime=np.append(filltime, filltime_p)
        lastpointLumi=np.append(lastpointLumi, lastpointLumi_p)
        true_time=np.append(true_time, true_time_p)
        ring_index = np.append(ring_index, ring_index_p)
        #append di date per diversi eta 
        Date = np.append(Date, date_fill)


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


    #redo sample weighting 
    weights_train1=[]
    for k in range (0, len(Norm_time_in_fill_train)):
        
        sample_weighting_dimension
        if Norm_time_in_fill_train[k] == 1. :
            weight = 10
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

    return [all_inputs_train, transp_train, weights_train_def, Norm_time_in_fill_train,Date]

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
    Date =[]
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
        date = [datetime.datetime.fromtimestamp(ts) for ts in metadata_test.time]
        #metadata_test = metadata_test[(metadata_test.lumi_inst >= 0.0001*1e9) & (metadata_test.lumi_inst <= 0.0004*1e9) & (metadata_test.lumi_in_fill >= 0.1*1e9)]
        transp_test = mean[metadata_test.index.values[0]:metadata_test.index.values[0]+len(metadata_test.axes[0])]
        fill_num_test = metadata_test.fill_num.unique()

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
                ring_index_test_p[j] = 1.479
            if m_key ==19:
                ring_index_test_p[j] = 1.566
            if m_key ==20:
                ring_index_test_p[j] = 1.653
            if m_key ==21:
                ring_index_test_p[j] = 1.740
            if m_key ==22:
                ring_index_test_p[j] = 1.830
            if m_key ==23:
                ring_index_test_p[j] = 1.930
            if m_key ==24:
                ring_index_test_p[j] = 2.043
            if m_key ==25:
                ring_index_test_p[j] = 2.172
            if m_key ==26:
                ring_index_test_p[j] = 2.322
            if m_key == 27:
                ring_index_test_p[j] = 2.500
            if m_key == 28:
                ring_index_test_p[j] = 2.650

        instLumi_test=np.append(instLumi_test, instLumi_test_p)
        intLumiLHC_test=np.append(intLumiLHC_test,intLumiLHC_test_p)
        infillLumi_test=np.append(infillLumi_test,infillLumi_test_p)
        lastfillLumi_test=np.append(lastfillLumi_test,lastfillLumi_test_P)
        filltime_test=np.append(filltime_test,filltime_test_p)
        lastpointLumi_test=np.append(lastpointLumi_test,lastpointLumi_test_p)
        true_time_test=np.append(true_time_test,true_time_test_p)
        ring_index_test=np.append(ring_index_test,ring_index_test_p)
        Date = np.append(Date, date)
        
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
    all_inputs_test = np.stack((instLumi_test, intLumiLHC_test, infillLumi_test, lastfillLumi_test, filltime_test, lastpointLumi_test), axis=-1)
    #k_fold_1 = np.stack((), axis=-1)
    #k_fold_2 = np.stack((), axis=-1)

    return [all_inputs_test, transp_test_final, weights_test, metadata_test, true_time_test, Norm_time_in_fill_test, fill_sizes, infillLumi_test,instLumi_test,Date]

#function that creates new validation data for cross validation
#def K_fold_cross_validation (dictionary_test, metadata): 
    

#---------------------------------------------"SMOOTHING"-------------------------------------------------------
#this for is used to hide this section


#--------Fill excluded from iring 22
nonsmooth22 = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5887, 5954, 5984, 6024, 
                6030, 6041, 6057, 6084, 6089, 6090, 6091, 6096, 6105, 6106, 6116,
                6152, 6159, 6160, 6167, 6168, 6192, 6193, 6263, 6318,

             #we exclude some fills for validation for iring 22
                 6356, 6358, 6360, 6362, 6364, 6370, 6371
                 ]

#--------Fill excluded from iring 23
nonsmooth23 = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5887, 5954, 5984, 6024, 
                6030, 6041, 6057, 6084, 6089, 6090, 6091, 6096, 6105, 6106, 6116,
                6152, 6159, 6160, 6167, 6168, 6192, 6193, 6263, 6318,

             #we exclude some fills for validation for iring 23
                #6356, 6358, 6360, 6362, 6364, 6370, 
                6371
                ]
nonsmooth23=[5697, 5698, 5699, 5710, 5719, 5736, 5739, 5740, 5749, 5859, 5860,
             5861, 5862, 5871, 5887, 5919, 5929, 5930, 5946, 5952, 5966, 5970,
             5971, 5974, 6001, 6005, 6006, 6015, 6018, 6021, 6034, 6039, 6047, 
             6055, 6072, 6130, 6132, 6146, 6155, 6164, 6173, 6179, 6180, 6183,
             6184, 6200, 6201, 6217, 6233, 6293, 6309, 6313, 6341, 6351, 6355,
             6374, 6382, 6387, 6388, 6390, 6402, 6431, 6432,
             

             6356, 6358, 6360, 6362, 6364, 6370, 6371
             #, 6343, 6346
             
             #K_FOLD_1: excluding fills for 1st fold cross validation 
             #6389, 6392, 6398
             #K_FOLD_2 : excluding fills for 2nd fold cross validation
             #6384, 6385, 6386, 6389
             
             #6415
             #6371
             #6272,
             #6295
             ]

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
             ]

#--------Fill excluded from iring 25
nonsmooth25 = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5883, 5887, 5954, 5980,
                5984, 6030, 6041, 6057, 6084, 6096, 6105, 6106, 6152, 
                6159, 6160, 6167, 6168, 6170, 6192, 6239, 6261,  6293, 6261, 6262, 6263, 6279, 
                6300, 6318, 6348, 6349,

                #we exclude some fills for validation for iring 26
               6356, 6358, 6360, 6362, 6364, 6370, 6371
                ]            
    #with this command we actually exclude nonsmooth fills from the "fill" array 
    #but we are also excluding validation fills which we insert in the validation dictionary (dictionary_test).

#--------Fill excluded from iring 26
nonsmooth26 = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5883, 5887, 5954, 5980,
                5984, 6030, 6041, 6057, 6084, 6096, 6105, 6106, 6116, 6119, 6152, 
                6159, 6160, 6167, 6168, 6170, 6171, 6192, 6261, 6262, 6263, 6279, 
                6300, 6318, 6348, 6349,

                #we exclude some fills for validation for iring 25
                6356, 6358, 6360, 6362, 6364, 6370, 6371
                ] 

#--------Fill excluded from iring 27
nonsmooth27 = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5883, 5887, 5954, 5980,
                5984, 6030, 6041, 6057, 6084, 6096, 6105, 6106, 6116, 6119, 6152, 
                6159, 6160, 6167, 6168, 6170, 6171, 6192, 6261, 6262, 6263, 6279, 
                6300, 6318, 6348, 6349,

                #we exclude some fills for validation for iring 25
                6356, 6358, 6360, 6362, 6364, 6370, 6371
                ] 

#--------Fill excluded from iring 28
nonsmooth28 = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5883, 5887, 5954, 5980,
                5984, 6030, 6041, 6057, 6084, 6096, 6105, 6106, 6116, 6119, 6152, 
                6159, 6160, 6167, 6168, 6170, 6171, 6192, 6261, 6262, 6263, 6279, 
                6300, 6318, 6348, 6349,

                #we exclude some fills for validation for iring 25
                6356, 6358, 6360, 6362, 6364, 6370, 6371
                ] 


#---------------------
#Fills used for inference

#here we can define various validation dataset, the last is the one to be used
filltest23 = [5958, 6031, 6046, 6053, 6110, 6324, 6356, 6371]

#validation single fill

#validaiton different fills 
filltest27 = [6356, 6358, 6360, 6362, 6364, 6370, 6371]
#filltest24 = [6356, 6358, 6360, 6362, 6364, 6370, 6371]

filltest24 = [6371#6272
               #6295
               ]

#filltest23 = [6371
                #6415
               # ]
filltest23 = [ 6343, 6346 ,

                6356, 6358, 6360, 6362, 6364, 6370, 6371#,
                #6384, 6385, 6386, 6389
                ] 




#----------------Cross validation fill for different i-Rings

#filltest23 = [6343, 6362, 6371] #i-Ring used for training 

filltest22 = [6343, 6362, 6371]
filltest24 = [6343, 6362, 6371]
filltest25 = [6343, 6362, 6371]
filltest26 = [6343, 6362, 6371]
filltest27 = [6343, 6362, 6371]
filltest28 = [6343, 6362, 6371]


filltest24 = [6343, 6346 , 6356, 6358, 6360, 6362, 6364, 6370, 6371]  #stessi fill usati nel test di eta23 trainando il modello con eta23
filltest22 = [6343, 6346 , 6356, 6358, 6360, 6362, 6364, 6370, 6371]  #stessi fill usati nel test di eta23 trainando il modello con eta23
filltest28 = [6343, 6346 , 6356, 6358, 6360, 6362, 6364, 6370, 6371]  #stessi fill usati nel test di eta23 trainando il modello con eta23


filltest23 = [5848, 6053 ,6140,  6174, 6191, 6243, 6275,6300,  6356]

#filltest23 = [5958, 6031, 6046, 6053, 6110, 6324, 6356, 6371]


filltest23 = [6140,  6174, 6191, 6243, 6110, 6324, 6356, 6371,5848]




filltest23 = [5848, 6053 ,6140,  6174, 6191, 6243, 
6275,  6300, 6356,
5958, 6031, 6046,]

#6053, 6110, 6324, 6356, 6371]


filltest23 = [ #5848, 6053 ,6140, 6174,6191,6243,

  #6275,
  #6300,
  #6356, 
  
  #5958, 
   #6031, 
  #6046, 

    #6053,
    #6110,
    #6324,
    #6371,
    #6298, 

    #  6318,      
    #  6343, 
    #  6346, 
    #  6358, 
    #  6362,
    #  6255, 
    #  6291, 
    #  6307,
    #  6314, 
     #6143

            #  5848, 6053 ,6140,  6174, 6191,
            #  6243, 

            # 6275, 
            # 6300, 
            # 6356,
            
            # 5958, 
            # 6031,
            # 6046,

            # 6053, 6110, 6324,
            
            # 6371, 
            # 6298,
]


fill_test_24 = [5704, 5717 ,5718, 5722, 5730, 5733, 5737, 5738, 5746, 5748, 5750, 5822, 5824, 5825,
  5830, 5833, 5834, 5837, 5838, 5839, 5840, 5842, 5845, 5848, 5849, 5856, 5864, 5865,
  5868, 5870 ,5872, 5873, 5874, 5876, 5878, 5880, 5882, 5883, 5885, 5920, 5933, 5934,
  5942, 5950, 5954, 5958, 5960, 5962, 5963, 5965, 5976, 5979, 5980, 5984 ,5985, 6012,
  6016, 6019, 6020, 6024, 6026, 6030, 6031, 6035, 6036, 6041, 6044, 6046 ,6048, 6050,
  6052 ,6053, 6054, 6057, 6060, 6061, 6079, 6082, 6084, 6086, 6089, 6090 ,6091, 6093,
  6094, 6096, 6097, 6098, 6104, 6105, 6106, 6110, 6114, 6116, 6119, 6123 ,6136, 6138,
  6140, 6141, 6142, 6143, 6147, 6152, 6156, 6158 ,6159, 6160, 6161, 6165 ,6167, 6168,
  6169, 6170, 6171 ,6174, 6175, 6176, 6177, 6182 ,6185, 6186, 6189, 6191 ,6192, 6193,
  6194, 6195, 6199, 6216, 6226, 6227, 6228, 6230 ,6231, 6232 ,6236, 6238 ,6239, 6240,
  6241, 6243, 6245, 6247, 6252, 6253, 6255, 6258 ,6259, 6261 ,6262, 6263 ,6266, 6268,
  6269 ,6271, 6272, 6275, 6276, 6278, 6279 ,6283 ,6284, 6285 ,6287, 6288 ,6291, 6294,
  6295, 6297, 6298, 6300, 6303, 6304, 6305, 6306 ,6307, 6308 ,6311, 6312 ,6314, 6315,
  6317, 6318, 6323, 6324, 6325, 6336, 6337, 6343 ,6344, 6346 ,6347, 6348 ,6349, 6377,
  6380, 6381, 6384, 6385, 6386, 6389, 6392, 6396 ,6397, 6398 ,6399, 6404 ,6405, 6411,
  6413 ,6415, 6417]
 

nonsmooth23=[]

#Train dictionaries used in the model that we load in this script
dictionary_train = { 23:["iRing23new.npy", nonsmooth23]} #no needed

#dictionary_test = {22:["iRing22new.npy", filltest22] }
dictionary_test = {28:["iRing28new.npy", filltest23] }
#dictionary_test = {24:["iRing24new.npy", filltest24] }
#dictionary_test = {28:["iRing28new.npy", filltest28] }

metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")

#Pre-processing data for training and validation
[all_inputs_train, transp_train, weights_train, norm_time_in_fill_train,date_train]=Pre_Processing_Train(dictionary_train, metadata)
[all_inputs_validation, transp_validation, weights_validation, metadata_test, time_test, norm_time_in_fill_test, fill_sizes, infilllumi_test,lumi_inst_validation, date_validation]=Pre_Processing_Test(dictionary_test,metadata)
#---------------------------------PLOTTING SOME COOL PLOTS


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



#single eta model DNN
inputs = Input(shape=(6,))
hidden1 = Dense(256, activation='leaky_relu', kernel_regularizer=l2(3e-7), bias_regularizer=regularizers.l2(3e-7),activity_regularizer=l2(3e-7))(inputs)
drop1=Dropout(1)(hidden1)
hidden2 = Dense(128, activation='leaky_relu', kernel_regularizer=l2(3e-7), bias_regularizer=regularizers.l2(3e-7),activity_regularizer=l2(3e-7))(drop1)
drop2=Dropout(1)(hidden2)
hidden3 = Dense(64, activation='leaky_relu', kernel_regularizer=l2(3e-7), bias_regularizer=regularizers.l2(3e-7),activity_regularizer=l2(3e-7))(drop2)
outputs = Dense(1) (hidden3)


#model checkpoint and early stopping
filepath = "/home/federico/root/root-6.24.06-install/weights_multiple_eta_model_nuovo.ckpt"


#separating plots for each fill
#here i can also calculate mse for eac fill and other metrics

#LOADING the model saved with the checpoint method 
#this makes sure that the best weights will be used to make inference
saved_model = Model ( inputs=inputs, outputs=outputs )
saved_model.load_weights(filepath)

saved_model.predict(all_inputs_validation)
#----------------------------------------

#actually  predicting transparency data 
transp_predicted_validation = saved_model.predict(all_inputs_validation)

#transp_predicted_train = saved_model.predict(all_inputs_train)

#transp_predicted_train = saved_model.predict(all_inputs_train)


plt.plot(date_validation,(transp_validation-transp_predicted_validation)/transp_validation, color='m',  markersize=2, linewidth =0,  label="predicted/measured", marker="p")
plt.xlabel('date[month/day/hour]')
plt.ylabel('\frac{measured}{predicted} mean transparency')
plt.legend()
plt.show()  

np.savetxt(f"{data_folder}/measured23.txt", transp_validation,fmt="%s ")
np.savetxt(f"{data_folder}/predicted23.txt", transp_predicted_validation,fmt="%s ")
np.savetxt(f"{data_folder}/lumi23.txt", all_inputs_validation, fmt="%s")

# np.savetxt(f"{data_folder}/lumi23.txt", all_inputs_train, fmt="%s")
# np.savetxt(f"{data_folder}/measured23.txt", transp_train, fmt="%s")
# np.savetxt(f"{data_folder}/predicted.txt", transp_predicted_train, fmt="%s")



ax1=plt.subplot(2,1,1)
plt.plot(date_validation,transp_validation,color='b',  markersize=2, linewidth =0.2,  label="predicted", marker="p")
plt.plot(date_validation,transp_predicted_validation,color='r',  markersize=0.2, linewidth =0,  label="predicted", marker="p")
plt.xticks(rotation='45')
ax1.get_ylabel()
ax1.set_ylabel("normalized mean transparency", color= 'black', fontweight='book', fontstyle='italic',fontsize='medium')
ax1.get_xlabel()
ax1.set_xlabel("date", color= 'black', fontweight='book', fontstyle='italic',fontsize='medium')

ax2=plt.subplot(2,1,2)
plt.plot(date_validation,infilllumi_test,color='b',  markersize=2, linewidth =0.2,  label="predicted", marker="p")
plt.xticks(rotation='45')
ax2.get_ylabel()
ax2.set_ylabel("lumi_inst", color= 'black', fontweight='book', fontstyle='italic',fontsize='large')
ax2.get_xlabel()
ax1.set_xlabel("date", color= 'black', fontweight='book', fontstyle='italic',fontsize='large')

#plt.plot(time_test,transp_predicted_validation,color='r',  markersize=2, linewidth =0,  label="predicted", marker="p")
plt.show()

#----------------------------------

#-----------Single Fill---------

selected_transp = transp_validation
print("trasparenza selezionata in questo fill")
print(selected_transp)
print(len(selected_transp))



#evaluating metrics and plotting

j=0
new_k = 0
counter_subplot = 1
m=0
sum_mse = 0
sum_mae = 0
#Time_test=[]
#legend


fig1 = plt.figure(figsize=(20,10))

for k in fill_sizes:

    Fill_Num = filltest23[m] 
    transp_predicted=[]
    transp_real=[]
    lumi_in_fill_test=[]
    Time_test=[]
    Date_test=[]

    new_k = new_k + k
    

    for i in range (int(j), int(new_k)):
        #print(norm_time_in_fill_train[i])
        transp_predicted_prov = transp_predicted_validation[i]
        transp_real_prov = transp_validation[i]
        Lumi_in_fill_prov_test =  infilllumi_test[i]
        time_prov_test = time_test[i]
        date_prov_test = date_validation[i]

        transp_predicted = np.append(transp_predicted, transp_predicted_prov)
        transp_real = np.append (transp_real, transp_real_prov)
        lumi_in_fill_test = np.append(lumi_in_fill_test, Lumi_in_fill_prov_test)
        Time_test=np.append(Time_test,time_prov_test)
        Date_test = np.append(Date_test, date_prov_test)

    j = new_k
    m=m+1
    
    #calculation of MSE for each fill of validation
    mse = ((transp_real-transp_predicted)**2)
    mse = sum (mse)/len(transp_real)
    print('mean squared error')
    print(mse)
    sum_mse = (sum_mse + mse)


    #calculation of MAE for each fill of validation (that gives a more strainghtforward idea of the error for each )
    mae = (abs(transp_real-transp_predicted))
    mae =  sum(mae)/len(transp_real)
    print('mean abs error')
    print(mae)
    sum_mae = (sum_mae+mae)
    

 






    
    #Plotting transparency for different fills - predicted vs real 

    ax = plt.subplot(3,3,counter_subplot)
    pred=ax.plot(Date_test, transp_predicted, color='r',  markersize=2, linewidth =0.2,  label="predicted", marker="^")
    real=ax.plot(Date_test, transp_real, color='b',  markersize=2, linewidth =0.2, label="measured", marker="s")
    
    plt.grid(linestyle = 'dashed')
    #ax = plt.subplot(2,3,counter_subplot+3)
    #lumi=ax.plot(Time_test, lumi_in_fill_test, "r--", markersize=2, linewidth =0, marker='p')
    #plt.xlabel("luminosity in fill $[fb^{-1}]$ ")
    if m==1:
        ax.get_ylabel()
        ax.set_ylabel("mean transparency", color= 'black', fontweight='book', fontstyle='italic',fontsize='large')
    if m==4:
        ax.get_ylabel()
        ax.set_ylabel("mean transparency", color= 'black', fontweight='book', fontstyle='italic',fontsize='large')
    if m==7:
        ax.get_ylabel()
        ax.set_ylabel("mean transparency", color= 'black', fontweight='book', fontstyle='italic',fontsize='large')
    if m==7:
        ax.get_xlabel()
        ax.set_xlabel("date", color= 'black', fontweight='book', fontstyle='italic',fontsize='large')
    if m==8:
        ax.get_xlabel()
        ax.set_xlabel("date", color= 'black', fontweight='book', fontstyle='italic',fontsize='large')
    if m==9:
        ax.get_xlabel()
        ax.set_xlabel("date", color= 'black', fontweight='book', fontstyle='italic',fontsize='large')
    
    # if m==1:
    #     ax.get_xlabel()
    #     ax.set_xlabel("integrated luminosity $[fb^{-1}]$", color= 'black', fontweight='book', fontstyle='italic',fontsize='large')
    # if m==2:
    #     ax.get_xlabel()
    #     ax.set_xlabel("integrated luminosity $[fb^{-1}]$", color= 'black', fontweight='book', fontstyle='italic',fontsize='large')
    # if m==3:
    #     ax.get_xlabel()
    #     ax.set_xlabel("integrated luminosity $[fb^{-1}]$", color= 'black', fontweight='book', fontstyle='italic',fontsize='large')
    
       # plt.ylabel("mean transparency", color= 'blue', fontweight='bold')


    plt.tick_params(labelsize=7)
    plt.xticks(rotation=45)
    #ax.get_title()
    #ax.set_title(f"fill_num = {Fill_Num} ", color= 'black', fontweight='book', fontstyle='normal',fontsize='large')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    white1 = mpatches.Patch(color='w', label=f"fill_num = {Fill_Num}")
    white2 = mpatches.Patch(color='w', label="$\eta=1.830$")

    blue = mpatches.Patch(color='b', label="real")
    orange = mpatches.Patch(color='r', label="predicted")
    eta = mpatches.Patch(edgecolor='black', facecolor=None, color=None, label='eta = 1.930 ')
    #labels = ax.get_legend_handles_labels()
    # if m==3:
    ax.legend([white1,white2],[f'{ Fill_Num}','i-Ring 23'])
    
    #ax.legend(loc='center', bbox_to_anchor=(0.5, -0.10), shadow=False, ncol=4)
    #plt.legend(['r', 'b', 'k', 'k'],(f"fill_num = {Fill_Num}", "$\eta=2.0430$", "predicted", "measured"))
    
    
    #plt.show()

    counter_subplot=counter_subplot+1


#plt.legend(handles=[blue, orange,eta], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

plt.subplots_adjust(left=0.049, bottom=0.095, right=0.551, top=0.97, wspace=0.148, hspace=0.092)
#ax.legend([blue, orange,eta],[ 'predicted transparency', 'real transparency'])
plt.show()

print('general mean square error for validation')
mse_validation = sum_mse/m
print(mse_validation)

print('general mean absolute error for validation')
mae_validation = sum_mae/m
print(mae_validation)

#---------------------------------------Plot luminosity ------------------------------------
j=0
new_k = 0
counter_subplot = 1
m=0

fig1 = plt.figure(figsize=(20,20))
relative_error=[]


#really carefull here

for k in fill_sizes:

    Fill_Num = filltest23[m] 
    transp_predicted=[]
    transp_real=[]
    lumi_in_fill_test=[]
    Time_test=[]
    new_k = new_k + k


    for i in range (int(j), int(new_k)):

        transp_predicted_prov = transp_predicted_validation[i]
        transp_real_prov = transp_validation[i]
        Lumi_in_fill_prov_test =  infilllumi_test[i]
        time_prov_test = time_test[i]

        transp_predicted = np.append(transp_predicted, transp_predicted_prov)
        transp_real = np.append (transp_real, transp_real_prov)
        lumi_in_fill_test = np.append(lumi_in_fill_test, Lumi_in_fill_prov_test)
        Time_test=np.append(Time_test,time_prov_test)
    j = new_k
    m=m+1
    #calculation of MSE for each fill of validation


    #calculation of relative error to plot
    relative_error = (transp_real-transp_predicted)/transp_real

    #plt.subplot(3,3,counter_subplot)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [Line2D([0], [0], color='b', lw=4, label='mse'),
                   Line2D([0], [0], marker='o', color='w', label='fill_num',
                          markerfacecolor='g', markersize=15),
                   Patch(facecolor='orange', edgecolor='r',
                         label='mae')]

    plt.subplot(3,3,counter_subplot)
    plt.plot(lumi_in_fill_test, Time_test , markersize=2, linewidth=0.75, label=(f'{Fill_Num}'), marker='p')
    plt.xlabel("time ")
    plt.ylabel("integrated luminosity $fb^{-1}$")
    plt.tick_params(labelsize=7)
    #plt.title(f"fill {Fill_Num}")
    
    plt.legend() 
    
    counter_subplot=counter_subplot+1

plt.show()
 # non toccare questa loss
history=np.load('my_history_weights_multiple_eta_model_nuovo.npy',allow_pickle='TRUE').item()

plt.plot( history["loss"], label = 'training loss function', color='green' )
plt.yscale("log")
plt.plot( history["val_loss"], label = 'validation loss function', color='crimson'  )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test
plt.xlabel("epochs ")
plt.ylabel("loss function")
plt.legend()
plt.grid(color='k', linestyle='--', linewidth=0.1)
plt.show()



#------------------------------------
#altre loss function 
history_noDropout = np.load('my_history_single_eta_model_noDropout.npy',allow_pickle='TRUE').item()

history_Dropout0 = np.load('my_history_single_eta_model_Dropout0.npy',allow_pickle='TRUE').item()


history_1 = np.load('my_history_single_eta_model_2.npy',allow_pickle='TRUE').item()

history_SGD = np.load('my_history_single_eta_model_SGD.npy',allow_pickle='TRUE').item()

history_L2e_6 = np.load('my_history_single_eta_model_L2e-6.npy',allow_pickle='TRUE').item()
history_L2e_8 = np.load('my_history_single_eta_model_L2e-8.npy',allow_pickle='TRUE').item()

#plot di varie loss function di train 
#plt.plot( history["loss"], label = 'Dropout 0.2 ' , color = 'b')


#plt.plot( history_noDropout["loss"], label = 'Dropout 0.1 Adam',color = 'lime' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test

plt.plot( history_Dropout0["loss"], label = 'Dropout 0.0 Adam $L_{2}=10^{-7}$',color = 'lime' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test

plt.plot( history_1["loss"], label = 'Dropout 0.2 Adam $ L_{2}=10^{-7}$',color = 'orange' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test

plt.plot( history_SGD["loss"], label = 'SGD optimizer',color = 'black' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test


plt.plot( history_L2e_6["loss"], label = 'Adam - $ L_{2}=10^{-6}$',linestyle='--',color = 'red' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test
plt.plot( history_L2e_8["loss"], label = 'Adam - $L_{2}=10^{-8}$',linestyle='--',color = 'cyan' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test

plt.yscale("log")
plt.ylabel("training loss function")
plt.xlabel("epochs ")
plt.legend()
plt.grid(color='k', linestyle='--', linewidth=0.1)
plt.show()

#plot di varie loss function di validation 
#plt.plot( history["val_loss"], label = 'Dropout 0.2 Adam' , color = 'b') 


#plt.plot( history_noDropout["val_loss"], label = 'Dropout 0.1 Adam Optimizer',color = 'lime' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test

plt.plot( history_Dropout0["val_loss"], label = 'Dropout 0.0 Adam $ L_{2}=10^{-7}$',color = 'lime' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test


plt.plot( history_1["val_loss"], label = 'Dropout 0.2 - Adam $ L_{2}=10^{-7}$',color = 'orange' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test

plt.plot( history_SGD["val_loss"], label = 'Dropout 0.2 - SGD optimizer',color = 'black' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test


plt.plot( history_L2e_6["val_loss"], label = 'Adam - $ L_{2}=10^{-6}$',linestyle='--',color = 'red' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test
plt.plot( history_L2e_8["val_loss"], label = 'Adam - $L_{2}=10^{-8}$',linestyle='--',color = 'cyan' )#devo inserire momenti senza radiazione per il train; non mi interessa usarli nel test

plt.yscale("log")
plt.ylabel("validation loss function ")
plt.xlabel("epochs ")
plt.legend()
plt.grid(color='k', linestyle='--', linewidth=0.1)
plt.show()



#plot of the model 
import visualkeras

visualkeras.layered_view(saved_model, to_file=f'{data_folder}/DNNPLOT.png', min_xy=10, min_z=10, scale_xy=100, scale_z=100, one_dim_orientation='x')

#explainability tools for machine learning 

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
    return saved_model.predict(x).flatten()

# compute the explainer. Chose Huber for its robustness against outliers
i=100
exp = explainer.explain_instance(all_inputs_validation[i,:],
                                  predict,
                                  num_features=6,
                                  distance_metric='euclidean',
                                 num_samples=len(all_inputs_validation),
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

