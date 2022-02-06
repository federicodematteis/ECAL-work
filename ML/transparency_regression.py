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
data23 = np.load(f"{data_folder}/iRing23new.npy")   #train
data23_df = pd.DataFrame(data23)
data23_df.head()
#iRing24
data24 = np.load(f"{data_folder}/iRing24new.npy")   #validation
data24_df = pd.DataFrame(data24)
data24_df.head()
#iring26
data26 = np.load(f"{data_folder}/iRing26new.npy")   #train
data26_df = pd.DataFrame(data26)
data26_df.head()

#transparency in iRings
mean23=[]
mean24=[]
mean26=[]

#compute mean transparnecy in iRings
for i in range (0, len(data23_df.axes[1])):
    #mean of colums' entries of data_df (for each "position" in the cristal)
    mean23 = np.append(mean23, np.mean(data23_df[i]))
    mean24 = np.append(mean24, np.mean(data24_df[i]))
    mean26 = np.append(mean26, np.mean(data26_df[i]))

mean23 = mean23[mean23 != -1]
mean24 = mean24[mean24 != -1]
mean26 = mean26[mean26 != -1]


#read metadata file 
metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")
metadata = metadata.iloc[:len(mean23)][mean23!=-1]
#timestamps for the entire run
date = [datetime.datetime.fromtimestamp(ts) for ts in metadata.time]

transp_norm_test=[]
transp = [mean24[i] for i in metadata.index.values]
transp = transp/transp[0]
transp_norm_test = np.append(transp_norm_test, transp)

transp_norm_train=[]
transp = [mean23[i] for i in metadata.index.values]
transp = transp/transp[0]
transp_norm_train = np.append(transp_norm_train, transp)
#secondo dataset training iRing26
transp_norm_train_2=[]
transp = [mean26[i] for i in metadata.index.values]
transp = transp/transp[0]
transp_norm_train_2 = np.append(transp_norm_train_2, transp)

#put metadata into arrays

#fill in iRings
fill = metadata["fill_num"].unique()
fill = fill[fill != 0]
metadata_fill = metadata[metadata.fill_num.isin(fill)]
#restrict dataset to the same as @brusale
#guardare meglio questa cosa( decidere se normalizzare)
metadata_fill = metadata_fill[(metadata_fill.lumi_inst >= 0.0001*1e9) & (metadata_fill.lumi_inst <= 0.0004*1e9) & (metadata_fill.lumi_in_fill >= 0.1*1e9)]
fill_num = metadata_fill.fill_num.unique()

#print(metadata_fill)
#print(metadata)
#print(fill_num)
metadata_duplicate = pd.concat([metadata, metadata])

instLumi = (1e-9)*metadata.loc[:,'lumi_inst']
intLumiLHC = (1e-9)*metadata.loc[:,'lumi_int']
infillLumi = (1e-9)*metadata.loc[:,'lumi_in_fill']
lastfillLumi = (1e-9)*metadata.loc[:,'lumi_last_fill']
filltime = (1e-9)*metadata.loc[:,'time_in_fill']
lastpointLumi = (1e-9)*metadata.loc[:, 'lumi_since_last_point']

#print(instLumi, infillLumi)

# print("dimensione di lumi_int")
# print(len(instLumi))
# print("dimensione di mean 25")
# print(len(mean23))

#put all inputs in one object
all_inputs=np.stack((instLumi, infillLumi, intLumiLHC, lastfillLumi, filltime, lastpointLumi), axis=-1)
#qui devo fare un append
#print(all_inputs)

#metadata_duplicate = pd.concat([metadata, metadata])

#print("metadata duplication")


#scatter plot mean transparency in iring lumi_in_fill-lumi_inst 
plt.figure()
plt.scatter(all_inputs[:,0], all_inputs[:, 1], c=mean23, cmap=plt.cm.RdBu, edgecolors='k')
plt.show()
plt.figure()

#shape degli input
inputs = Input(shape=(6,))
#first layer con 500 neuroni, e f=relu
hidden = Dense(500, activation='relu')(inputs)
#second layer con 100 neuroni e f=relu
hidden = Dense(100, activation='relu')(inputs)


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
#N = len(instLumi)
#nsplit=int(N/2)
all_inputs_training   = all_inputs
all_inputs_validation = all_inputs
transp_training   = transp_norm_train
transp_validation = transp_norm_test


print(all_inputs_validation)
# now actually performing the train 
history = model.fit( all_inputs_training, transp_training, validation_data = (all_inputs_validation,transp_validation), epochs=150, verbose=0)

# ... and plot the training loss
plt.plot( history.history["val_loss"] )
plt.plot( history.history["loss"] )
plt.show()


# now test the performance of the DNN
transp_predicted_validation = model.predict(all_inputs_validation)


plt.plot(metadata.time, transp_validation, "b .")
plt.plot(metadata.time, transp_predicted_validation, "r +")

plt.show()

print(transp_predicted_validation)

#problema, non predice la crescita di trasparenza dovuta alla morte dei centri di colore