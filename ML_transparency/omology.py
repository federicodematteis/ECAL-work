
from itertools import tee
#from tkinter.ttk import LabelFrame
import matplotlib.pyplot as plt
#import ROOT
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
#Libraries for machine learning
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, LeakyReLU, Add, Concatenate, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils  import plot_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
import seaborn as sns
# usefull tools for dimensionality reduction.
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import umap

  
#data_folder = ('/home/federico/root/root-6.24.06-install')
#data_folder = ('/gwpool/users/fdematteis')

metadata = pd.read_csv("fill_metadata_2017_10min.csv")

#----------------------------------------------------------
#transparency data for a selected iRing#

#23
data24=np.load("iRing23new.npy")
data24_df = pd.DataFrame(data24)
data24_df.head()
mean24=[]
#25
data25=np.load("iRing25new.npy")
data25_df = pd.DataFrame(data25)
data25_df.head()
mean25=[]

#compute mean transparnecy in iRings 
for i in range (0, len(data24_df.axes[1])):
    #mean of colums' entries of data_df (for each "position" in the cristal)
    mean24 = np.append(mean24, np.mean(data24_df[i]))

for i in range (0, len(data25_df.axes[1])):
    mean25 = np.append(mean25, np.mean(data25_df[i]))
#iring23
mean24=mean24[mean24 != -1]
metadata = metadata.iloc[:len(mean24)][mean24 != -1]
#iring25
mean25=mean25[mean25 != -1]
metadata1 = metadata.iloc[:len(mean25)][mean25 != -1]

#selecting metadata for fill (locking metadata to in_fill=1)
fill=metadata["fill_num"].unique()
fill = fill[fill != 0]

fill1=metadata1["fill_num"].unique()
fill1 = fill1[fill1 != 0]

nonsmooth = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5887, 5954, 5984, 6024, 
             6030, 6041,
             6057, 6084, 6089, 6090, 6091, 6096, 6105, 6106, 6116,
             6152, 6159, 6160, 6167, 6168,
             #6253, 6255, 
             6192, 6193, #6262,
             6263, 6318
             #6279 # non-smooth fills : 22/138
             #6349
#--------second tranche
             ]
nonsmooth1 = [5830, 5837, 5839, 5840, 5842, 5864, 5882, 5883, 5887, 5954, 5980,
              5984,  6030, 6041, 6057, 6084, 6096, 6105, 6106, 6116, 6119, 6152, 6159,
              6160, 6167, 6168, 6170, 6171, 6192, 6261, 6262, 6263, 6279, 6300, 6318, 6348,
              6349
              ]
             
for iev in range (0, len(nonsmooth)) :
    #print(nonsmooth[iev])
    fill = fill[fill != nonsmooth[iev]]

for iev in range (0, len(nonsmooth1)) :
    #print(nonsmooth[iev])
    fill1 = fill1[fill1 != nonsmooth1[iev]]

#test fill
#escludo fill di test
fill=fill[fill != 6371]
metadata_fill = metadata[metadata.fill_num.isin(fill)]
metadata_fill = metadata_fill[(metadata_fill.lumi_inst >= 0.0001*1e9) & (metadata_fill.lumi_inst <= 0.0004*1e9) & (metadata_fill.lumi_in_fill >= 0.1*1e9)]
fill_num = metadata_fill.fill_num.unique()

metadata_fill1 = metadata[metadata.fill_num.isin(fill1)]
metadata_fill1 = metadata_fill1[(metadata_fill1.lumi_inst >= 0.0001*1e9) & (metadata_fill1.lumi_inst <= 0.0004*1e9) & (metadata_fill1.lumi_in_fill >= 0.1*1e9)]
fill_num1 = metadata_fill1.fill_num.unique()


#print("fills usati per il train")
#print(fill_num)

transp_fill = []
transp_fill1 = []

lumi_inst_0 = []
lumi_int_0 = []

#riempie il vettore di transparency

#main for ------
for k in fill_num:
#transparency relativa ai fill selezionati
    df = metadata_fill[metadata_fill.fill_num == k]
    transp = [mean24[i] for i in df.index.values]
    #transp ha la grandezza del dataframe ristretto al k esimo fill
    transp = transp/transp[0]
    transp_fill = np.append(transp_fill, transp)
    a = np.empty(np.size(transp))
    b = np.empty(np.size(transp))
    a.fill(df['lumi_inst'].iloc[0])
    b.fill(df['lumi_int'].iloc[0])
    lumi_inst_0 = np.append(lumi_inst_0, a)
    lumi_int_0 = np.append(lumi_int_0, b)
    #in transp_fill ci sono i dati di trasparenza normalizzata per ogni fill;
#-------
for k in fill_num1:
    df1 = metadata_fill1[metadata_fill1.fill_num == k]
    transp1 = [mean25[i] for i in df1.index.values]
    transp1 = transp1/transp1[0]
    transp_fill1 = np.append(transp_fill1, transp1)

#-----iring23
instLumi = (1e-9)*metadata_fill.loc[:,'lumi_inst']
intLumiLHC = (1e-9)*metadata_fill.loc[:,'lumi_int']
infillLumi = (1e-9)*metadata_fill.loc[:,'lumi_in_fill']
lastfillLumi = (1e-9)*metadata_fill.loc[:,'lumi_last_fill']
filltime = (1e-9)*metadata_fill.loc[:,'time_in_fill']
lastpointLumi = (1e-9)*metadata_fill.loc[:, 'lumi_since_last_point']
true_time = (1e-9)*metadata_fill.loc[:, 'time']

ring_index = np.zeros(len(metadata_fill))
for i in range (0, len(ring_index)):
    ring_index[i] = 23
    
#-----iring25
instLumi1 = (1e-9)*metadata_fill1.loc[:,'lumi_inst']
intLumiLHC1 = (1e-9)*metadata_fill1.loc[:,'lumi_int']
infillLumi1 = (1e-9)*metadata_fill1.loc[:,'lumi_in_fill']
lastfillLumi1 = (1e-9)*metadata_fill1.loc[:,'lumi_last_fill']
filltime1 = (1e-9)*metadata_fill1.loc[:,'time_in_fill']
lastpointLumi1 = (1e-9)*metadata_fill1.loc[:, 'lumi_since_last_point']
true_time1 = (1e-9)*metadata_fill1.loc[:, 'time']

ring_index1 = np.zeros(len(metadata_fill1))
for i in range (0, len(ring_index1)):
    ring_index1[i] = 25

#now i have to merge transparency datas and luminosity metadatas
#metadata merge
merged_instLumi = np.append(instLumi, instLumi1)
merged_intLumiLHC = np.append(intLumiLHC, intLumiLHC1)   
merged_infillLumi = np.append(infillLumi, infillLumi1)
merged_lastfillLumi = np.append(lastfillLumi, lastfillLumi1)
merged_filltime = np.append(filltime, filltime1)
merged_ring_index = np.append(ring_index, ring_index1)
#transparency merge 

transp_train = np.append(transp_fill, transp_fill1)
all_inputs_train=np.stack((merged_instLumi, merged_infillLumi, merged_intLumiLHC, merged_filltime, merged_ring_index, merged_lastfillLumi), axis=-1)

#validation dataset
filltest = 6371
metadata_6371 = metadata[metadata.fill_num == filltest]
metadata_6371 = metadata_6371[(metadata_6371.lumi_inst >= 0.0001*1e9) & (metadata_6371.lumi_inst <= 0.0004*1e9) & (metadata_6371.lumi_in_fill >= 0.1*1e9)]
transp_6371 = mean24[metadata_6371.index.values[0]:metadata_6371.index.values[0]+len(metadata_6371.axes[0])]

# normalizzo il dato di trasparenza a quella precedente al fill
transp_6371 = transp_6371/transp_6371[0]

instLumi_test = (1e-9)*metadata_6371.loc[:,'lumi_inst']
intLumiLHC_test = (1e-9)*metadata_6371.loc[:,'lumi_int']
infillLumi_test = (1e-9)*metadata_6371.loc[:,'lumi_in_fill']
lastfillLumi_test = (1e-9)*metadata_6371.loc[:,'lumi_last_fill']
filltime_test = (1e-9)*metadata_6371.loc[:,'time_in_fill']
lastpointLumi_test = (1e-9)*metadata_6371.loc[:, 'lumi_since_last_point']
true_time_test = (1e-9)*metadata_6371.loc[:, 'time']

ring_index_test = np.zeros(len(metadata_6371))
for iev in range (0, len(metadata_6371)):
    ring_index_test[iev] = 23

all_inputs_test=np.stack((instLumi_test, infillLumi_test, intLumiLHC_test, filltime_test, ring_index_test,lastfillLumi_test), axis=-1)



#riduzione di dimensionalità 
#initializing umap parameters:

n_neighbors=15 
min_dist=0.1 
n_components=2
metric='euclidean'

reducer= umap.UMAP( n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric)

# penguin_data = penguins[
#     [
#         "culmen_length_mm",
#         "culmen_depth_mm",
#         "flipper_length_mm",
#         "body_mass_g",
#     ]
# ].values

scaled_train_dataset = StandardScaler().fit_transform(all_inputs_train)

#embedding 
embedding = reducer.fit_transform(scaled_train_dataset)
embedding.shape
print(embedding.shape)
#calculate persistent omology
# from gtda.homology import VietorisRipsPersistence
# VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2]) # Parameter explained in the text
# diagrams = VR.fit_transform(all_inputs_test)
# diagrams.shape
# #persistent diagram
# from gtda.plotting import plot_diagram
# i = 0
# plot_diagram(diagrams[i])

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=transp_train,  cmap=plt.cm.RdBu, edgecolors='k'
    )
plt.gca().set_aspect('equal', 'datalim')
plt.title('Luminosity Dataset Embedding', fontsize=18)
plt.show()

#------------Omology-----------
from gtda.homology import VietorisRipsPersistence

""" Connectivity information
0-dimensional homology β0 or H0, measures clusters; 
1-dimensional homology1 β1 or H1, measures loops; and
2- dimensional homology β2 or H2, measures voids (empty spaces) """
homology_dimensions = [0, 1, 2]
#homology_dimensions = [0, 1, 2]
VR = VietorisRipsPersistence(
                        homology_dimensions=homology_dimensions,
                        coeff=3,
                        n_jobs=-1)

diagrams =VR.fit_transform(np.array(embedding))
#diagram_0 =VR.fit_transform(np.array(intermediate_output_0)[None, : , :])
diagrams.shape
print('hello')
from gtda.plotting import plot_diagram

i = 0
plot_diagram(diagrams[i])

#UMAP algorithm has differents parameters affecting the resulting embedding


# def draw_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
#     fit = umap.UMAP(
#         n_neighbors=n_neighbors,
#         min_dist=min_dist,
#         n_components=n_components,
#         metric=metric
#     )
#     u = fit.fit_transform(scaled_train_dataset)
#     fig = plt.figure()
#     if n_components == 1:
#         ax = fig.add_subplot(111)
#         ax.scatter(u[:,0], range(len(u)), c=transp_train)
#     if n_components == 2:
#         ax = fig.add_subplot(111)
#         ax.scatter(u[:,0], u[:,1], c=transp_train)
#     if n_components == 3:
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(u[:,0], u[:,1], u[:,2], c=transp_train, s=100)
#     plt.title(title, fontsize=18)

# for n in (2, 5, 10, 20, 50, 100, 200):
#     draw_umap(n_neighbors=n, title='n_neighbors = {}'.format(n))
