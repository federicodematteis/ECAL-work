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
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils  import plot_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import seaborn as sns

data_folder=("/home/federico/root/root-6.24.06-install")
metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")
data = np.load(f"{data_folder}/iRing23new.npy")
data_df = pd.DataFrame(data)

mean = []          #train-iRing
for i in range(0, len(data_df.axes[1])):
    mean = np.append(mean, np.mean(data_df[i]))  

mean = mean[mean != -1]

metadata = metadata.iloc[:len(mean)][mean != -1]


fill = metadata["fill_num"].unique()
fill = fill[fill != 0]
#locking metadata file
metadata_fill = metadata[metadata.fill_num.isin(fill)]
metadata_fill = metadata_fill[(metadata_fill.lumi_inst >= 0.0001*1e9) & (metadata_fill.lumi_inst <= 0.0004*1e9) & (metadata_fill.lumi_in_fill >= 0.1*1e9)]
fill_num = metadata_fill.fill_num.unique()

transp_fill = []
lumi_inst_0 = []
lumi_int_0 = []

#transparency for all fill_num

for k in fill_num:
    df = metadata_fill[metadata_fill.fill_num == k]
    transp = [mean[i] for i in df.index.values]
    transp = transp/transp[0]
    transp_fill = np.append(transp_fill, transp)
    #plot transparency in fill k vs absolute time
    plt.figure()
    plt.plot(df.time, transp, ".b-", markersize=3, linewidth=0.75, label="Observed")
    plt.xlabel("time")
    plt.ylabel("Normalized mean transparency in fill")
    plt.tick_params(labelsize=7)
    plt.title(f"fill num {k}")

    plt.show()
    a = np.empty(np.size(transp))
    b = np.empty(np.size(transp))
    a.fill(df['lumi_inst'].iloc[0])
    b.fill(df['lumi_int'].iloc[0])
    lumi_inst_0 = np.append(lumi_inst_0, a)
    lumi_int_0 = np.append(lumi_int_0, b)






    
