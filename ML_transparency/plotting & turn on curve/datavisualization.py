from cmath import exp
from itertools import tee
from tkinter.ttk import LabelFrame
import matplotlib.pyplot as plt
import ROOT
import matplotlib as mpl
import pandas as pd
import numpy as np

#libraries for data analysis
from pprint import pprint
from collections import namedtuple
import datetime
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from sklearn.metrics import mean_squared_error
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.backend import get_session


data_folder = ('/home/federico/root/root-6.24.06-install')

metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")

#23
data23=np.load(f"{data_folder}/iRing23new.npy")
data23_df = pd.DataFrame(data23)
data23_df.head()
mean23=[]

for i in range (0, len(data23_df.axes[1])):
    #mean of colums' entries of data_df (for each "position" in the cristal)
    mean23 = np.append(mean23, np.mean(data23_df[i]))


#iring23

mean23=mean23[mean23 != -1]
metadata = metadata.iloc[:len(mean23)][mean23 != -1]

#selecting metadata for fill (locking metadata to in_fill=1)
fill=metadata["fill_num"].unique()
fill = fill[fill != 0]

#fill esclusi dal train
nonsmooth = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5887, 5954, 5984, 6024, 
             6030, 6041,
             6057, 6084, 6089, 6090, 6091, 6096, 6105, 6106, 6116,
             6152, 6159, 6160, 6167, 6168,
             6192, 6193, 
             6263, 6318,
             #escludo anche i fill che userò per il train
             6324, 6371, 6031, 6356, 6053, 5958, 6110, 6046]

#fill usati per il test             
nonsmooth_test= [6324, 6371, 6031, 6356, 6053, 5958, 6110, 6046]

for iev in range (0, len(nonsmooth)) :
    #print(nonsmooth[iev])
    fill = fill[fill != nonsmooth[iev]]
             

#ora escludo i fill che ho deselezionato dai dati di train
#i fill di validation sono gia stati esclusi
metadata_fill = metadata[metadata.fill_num.isin(fill)]
metadata_fill = metadata_fill[(metadata_fill.lumi_inst >= 0.0001*1e9) & (metadata_fill.lumi_inst <= 0.0004*1e9) & (metadata_fill.lumi_in_fill >= 0.1*1e9)]
fill_num = metadata_fill.fill_num.unique()

Time_in_fill = []
Tprov=[]
transp_fill = []
initial_time_in_fill=[]

for k in fill_num:
#transparency relativa ai fill selezionati
    df = metadata_fill[metadata_fill.fill_num == k]
    transp = [mean23[i] for i in df.index.values]
    #transp ha la grandezza del dataframe ristretto al k esimo fill
    transp = transp/transp[0]
    transp_fill = np.append(transp_fill, transp)


    #voglio selezionare il tempo iniziale di ogni fill
    t_0 = np.empty(np.size(transp))
    t_0.fill(df['time_in_fill'].iloc[0])
    #print('this')
    #print(df['time_in_fill'].iloc[0])
    #printa tutto time in fill per un fill preciso(6308)
    
    #ora creo un nuovo vettore in cui salvo tutti i time in fill dei fill usati nel train
    initial_time_in_fill = np.append(initial_time_in_fill, t_0)
    Time_in_fill = np.append(Time_in_fill, Tprov)
    #in transp_fill ci sono i dati di trasparenza normalizzata per ogni fill;
#print('time_in_fill[0] di tuuti i fill usati per il train')
# print(initial_time_in_fill)
# print(len(initial_time_in_fill))

#io però sono interessato ai time_in_fill[0] delle validation, comunque è una prova

#TRAIN
Norm_time_in_fill_train=[]
Norm_time_in_fill_prov_train=[]

for k in fill_num:
    df = metadata_fill[metadata_fill.fill_num == k]
    
    #time_in_fill normalizzato per ogni fill, rispetto all'istante iniziale
    Norm_time_in_fill_prov_train = metadata_fill.loc[:,'time_in_fill']
    Norm_time_in_fill_prov_train = Norm_time_in_fill_prov_train/df['time_in_fill'].iloc[0]
    Norm_time_in_fill_train=np.append(Norm_time_in_fill_train,Norm_time_in_fill_prov_train)
    print('time in fill normalizzato')
    print(Norm_time_in_fill_train)
    #print(len(Norm_time_in_fill))

#validation
Norm_time_in_fill_test=[]
Norm_time_in_fill_prov_test=[]

for k in fill_num:
    df = metadata_fill[metadata_fill.fill_num == k]
    
    #time_in_fill normalizzato per ogni fill, rispetto all'istante iniziale
    Norm_time_in_fill_prov_test = metadata_fill.loc[:,'time_in_fill']
    Norm_time_in_fill_prov_test = Norm_time_in_fill_prov_test/df['time_in_fill'].iloc[0]
    Norm_time_in_fill_test=np.append(Norm_time_in_fill_test,Norm_time_in_fill_prov_test)
    print('time in fill normalizzato')
    print(Norm_time_in_fill_test)






