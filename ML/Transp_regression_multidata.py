from itertools import tee
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
#Libraries for machine learning
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, LeakyReLU, Add, Concatenate, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.utils  import plot_model
from tensorflow.keras.layers import Layer

data_folder = ('/home/federico/root/root-6.24.06-install')

#load dtaa from numpy arrays
#iRing23
data23 = np.load(f"{data_folder}/iRing23new.npy")   #train1
data23_df = pd.DataFrame(data23)
data23_df.head()
#iRing24
data24 = np.load(f"{data_folder}/iRing24new.npy")   #validation
data24_df = pd.DataFrame(data24)
data24_df.head()
#iring26
data26 = np.load(f"{data_folder}/iRing26new.npy")   #train2
data26_df = pd.DataFrame(data26)
data26_df.head()
#iRing25 
data25 = np.load(f"{data_folder}/iRing25new.npy")   #train3
data25_df = pd.DataFrame(data25)
data25_df.head()

#transparency in iRings
mean23=[]
mean24=[]
mean26=[]
mean25=[]

#compute mean transparnecy in iRings
for i in range (0, len(data23_df.axes[1])):
    #mean of colums' entries of data_df (for each "position" in the cristal)
    mean23 = np.append(mean23, np.mean(data23_df[i]))
    mean24 = np.append(mean24, np.mean(data24_df[i]))
    mean26 = np.append(mean26, np.mean(data26_df[i]))
    mean25 = np.append(mean25, np.mean(data25_df[i]))
mean23 = mean23[mean23 != -1]
mean24 = mean24[mean24 != -1]
mean26 = mean26[mean26 != -1]
mean25 = mean25[mean25 != -1]

#read metadata file 
metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")
metadata = metadata.iloc[:len(mean23)][mean23!=-1]
#metadata = metadata[(metadata.lumi_inst >= 0.0001*1e9) & (metadata.lumi_inst <= 0.0004*1e9) & (metadata.lumi_in_fill >= 0.1*1e9)]

#timestamps for the entire run
date = [datetime.datetime.fromtimestamp(ts) for ts in metadata.time]

#dataset training iRing24
transp_norm_test=[]
transp = [mean24[i] for i in metadata.index.values]
transp = transp/transp[0]
transp_norm_test = np.append(transp_norm_test, transp)
#dataset training iRing23
transp_norm_train=[]
transp = [mean23[i] for i in metadata.index.values]
transp = transp/transp[0]
transp_norm_train = np.append(transp_norm_train, transp)

# dataset validation iRing26
transp_norm_train_2=[]
transp = [mean26[i] for i in metadata.index.values]
transp = transp/transp[0]
transp_norm_train_2 = np.append(transp_norm_train_2, transp)

# dataset validation iRing26
transp_norm_train_3=[]
transp = [mean25[i] for i in metadata.index.values]
transp = transp/transp[0]
transp_norm_train_3 = np.append(transp_norm_train_3, transp)

transp_norm_TRAIN = np.append(transp_norm_train, transp_norm_train_2)
transp_norm_TRAIN2 = np.append(transp_norm_TRAIN, transp_norm_train_3)

#print("trasparenza dei due eta-rings concatenata")
#print(transp_norm_TRAIN)
#print(len(transp_norm_TRAIN))

#❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉

#fill in iRings
fill = metadata["fill_num"].unique()
fill = fill[fill != 0]
metadata_fill = metadata[metadata.fill_num.isin(fill)]
#restrict dataset to the same as @brusale
metadata_fill = metadata_fill[(metadata_fill.lumi_inst >= 0.0001*1e9) & (metadata_fill.lumi_inst <= 0.0004*1e9) & (metadata_fill.lumi_in_fill >= 0.1*1e9)]
fill_num = metadata_fill.fill_num.unique()

#❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉❆❅❉

print(metadata)

#put metadata into arrays
instLumi = (1e-9)*metadata.loc[:,'lumi_inst']
intLumiLHC = (1e-9)*metadata.loc[:,'lumi_int']
infillLumi = (1e-9)*metadata.loc[:,'lumi_in_fill']
lastfillLumi = (1e-9)*metadata.loc[:,'lumi_last_fill']
filltime = (1e-9)*metadata.loc[:,'time_in_fill']
lastpointLumi = (1e-9)*metadata.loc[:, 'lumi_since_last_point']
true_time = (1e-9)*metadata.loc[:, 'time']

#...ᘛ⁐̤ᕐᐷ ...ᘛ⁐̤ᕐᐷ ...ᘛ⁐̤ᕐᐷ ...ᘛ⁐̤ᕐᐷ ...ᘛ⁐̤ᕐᐷ ...ᘛ⁐̤ᕐᐷ ...ᘛ⁐̤ᕐᐷ ...ᘛ⁐̤ᕐᐷ

#creating new arrays for the new training metadata                              
instLumi_duplicate = (1e-9)*metadata.loc[:,'lumi_inst']                             
merged_instLumi_duplicate = np.append(instLumi, instLumi_duplicate)
merged_instLumi_duplicate2 = np.append(instLumi, merged_instLumi_duplicate)

intLumiLHC_duplicate = (1e-9)*metadata.loc[:,'lumi_int']                                                  
merged_intLumiLHC_duplicate = np.append(intLumiLHC, intLumiLHC_duplicate)   
merged_intLumiLHC_duplicate2 = np.append(intLumiLHC, merged_intLumiLHC_duplicate)   

infillLumi_duplicate = (1e-9)*metadata.loc[:,'lumi_in_fill']
merged_infillLumi_duplicate = np.append(infillLumi, infillLumi_duplicate)
merged_infillLumi_duplicate2 = np.append(infillLumi, merged_infillLumi_duplicate)

lastfillLumi_duplicate = (1e-9)*metadata.loc[:,'lumi_last_fill']
merged_lastfillLumi_duplicate = np.append(lastfillLumi, lastfillLumi_duplicate)
merged_lastfillLumi_duplicate2 = np.append(lastfillLumi, merged_lastfillLumi_duplicate)

filltime_duplicate = (1e-9)*metadata.loc[:,'time_in_fill']
merged_filltime_duplicate = np.append(filltime, filltime_duplicate)
merged_filltime_duplicate2 = np.append(filltime, merged_filltime_duplicate)

true_time_duplicate = (1e-9)*metadata.loc[:,'time']
merged_true_time_duplicate = np.append(true_time, true_time_duplicate)
merged_true_time_duplicate2 = np.append(true_time, merged_true_time_duplicate)

lastpointLumi_duplicate = (1e-9)*metadata.loc[:,'lumi_since_last_point']
merged_lastpointLumi_duplicate = np.append(lastpointLumi, lastpointLumi_duplicate)
merged_lastpointLumi_duplicate2 = np.append(lastpointLumi, merged_lastpointLumi_duplicate)

#   (▀̿Ĺ̯▀̿ ̿)--------|̲̅̅●̲̅̅|̲̅̅=̲̅̅|̲̅̅●̲̅̅|--------ヽ(⌐■_■)ノ♪♬

#add all inputs into one object
#test
all_inputs=np.stack((instLumi, infillLumi, intLumiLHC, lastfillLumi, filltime, lastpointLumi), axis=-1)
#train
all_inputs_duplicate=np.stack((merged_instLumi_duplicate2, merged_infillLumi_duplicate2, merged_intLumiLHC_duplicate2, merged_lastfillLumi_duplicate2, merged_filltime_duplicate2, merged_lastpointLumi_duplicate2), axis=-1)

#Scatter plot mean transparency in iring lumi_in_fill-lumi_inst 
plt.figure()
plt.scatter(all_inputs[:,0], all_inputs[:, 1], c=mean23, cmap=plt.cm.RdBu, edgecolors='k')
plt.show()
plt.figure()

# Machine learning  ─=≡Σ(([ ⊐•̀⌂•́]⊐

#shape degli input
inputs = Input(shape=(6,))
#first layer con 500 neuroni, and f=relu
hidden = Dense(500, activation='leaky_relu')(inputs)
#second layer with 100 neurons and f=relu
hidden = Dense(100, activation='leaky_relu')(inputs)
#third layer with 50 neurons and f=relu
#hidden = Dense(10, activation='leaky_relu')(inputs)

outputs = Dense(1)(hidden)
model = Model ( inputs=inputs, outputs=outputs )
model.compile( 
     loss='MSE',
     optimizer='adam'
     )

#write the summary of the network
model.summary()

# plot the network
plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
)

all_inputs_training   = all_inputs_duplicate
all_inputs_validation = all_inputs
transp_training   = transp_norm_TRAIN2
transp_validation = transp_norm_test

print(all_inputs_validation)
# now actually performing the train (ง •̀_•́)ง
#da errore sul train perchè vede 54mila vs 
history = model.fit( all_inputs_training, transp_training, validation_data = (all_inputs_validation,transp_validation), epochs=150, verbose=0)

# ... and plot the training loss
plt.plot( history.history["val_loss"] )
plt.plot( history.history["loss"] )
plt.show() 

# now test the performance of the DNN ಠ_ರೃ

transp_predicted_validation = model.predict(all_inputs_validation)

plt.plot(metadata.time, transp_validation, "b .")
plt.plot(metadata.time, transp_predicted_validation, "r +")
plt.show()
print("Predicted mean Transparency in iRing")
print(transp_predicted_validation)
  
#problema, non predice la crescita di trasparenza dovuta alla morte dei centri di colore

