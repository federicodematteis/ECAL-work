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
from sklearn.preprocessing import MinMaxScaler

data_folder=("/home/federico/root/root-6.24.06-install")
metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")
data = np.load(f"{data_folder}/iRing23new.npy")
data_df = pd.DataFrame(data)

mean = []          #train-iRing

for i in range(0, len(data_df.axes[1])):
    mean = np.append(mean, np.mean(data_df[i]))  

mean = mean[mean != -1]

metadata = metadata.iloc[:len(mean)][mean != -1]

print(metadata)
fill = metadata["fill_num"].unique()
print(fill)
fill = fill[fill != 0]

nonsmooth23=[ #5697, 5698, 5699, 5710, 5719, 5733, 5736, 5737, 5738,5739, 5740, 5748,5749, 
#              5833,  5834, 5837,       5859, 5860,
#              5861, 5862, 5870, 5871, 5873, 5887, 5919, 5920, 5929, 5930, 5946, 5952, 5966, 5970,
#              5971, 5974, 6001, 6005, 6006, 6015, 6018, 6021, 6034, 6039, 6044, 6047, 
#              6055, 6072, 6130, 6132, 6146, 6155, 6164, 6173, 6179, 6180, 6183,
#              6184, 6195, 6199, 6200, 6201, 6217, 6226, 6227, 6228, 6230, 6331, 6232, 6233, 6238,
#              6293, 6309, 6313, 6336, 6341, 6351, 6355,
#              6374, 6382, 6387, 6388, 6390, 6402, 6431, 6432,
             
#             5704, 5722, 5822, 5824, 5825, 5830, 5838,5874, 5885, 5962, 5963, 5965,
#             5984, 5985, 6041, 6050, 6093, 6094, 6105, 6185, 6236,


#             #vecchi fill
#             # 6356, 6358, 6360, 6362, 6364, 6371,
#              #k-fold cross validation fills
#              #6370, 6343, 6346

#             #nuovi fill smooth usati per validation
#             5848, 6053 ,6140,  6174, 6191, 6243, 6275, 6300, 6356,
#             5958, 6031, 6046, 6053, 6110, 6324, 6356, 6371, 6141, 6232,
#             6261, 6263, 6269, 6279, 6294, 6377, 6380, 6381, 6386, 6404, 6405
#             #misto tra fill buini e non buoni
#             #6140,  6174, 6191, 6243, 6110, 6324, 6356, 6371,5848
#             #fill cattivi 
#             #5958, 6031, 6046, 6053, 6110, 6324, 6356, 6371

#             #6258, 
#               
]

for iev in range (0, len(nonsmooth23)):
     fill = fill [fill != nonsmooth23[iev]]

#locking metadata file
metadata_fill = metadata[metadata.fill_num.isin(fill)]
#metadata_fill = metadata_fill[(metadata_fill.lumi_inst >= 0.0001*1e9) & (metadata_fill.lumi_inst <= 0.0004*1e9) & (metadata_fill.lumi_in_fill >= 0.1*1e9)]
fill_num = metadata_fill.fill_num.unique()

transp_fill = []
lumi_inst_0 = []
lumi_int_0 = []
Date = []
#transparency relativa a tutti gli iring

for k in fill_num:
    df = metadata_fill[metadata_fill.fill_num == k]
    transp = [mean[i] for i in df.index.values]

    # #rescaling transparency target function
    # max_transp_fill = max(transp)
    # min_transp_fill=min(transp)
    # transp = (transp-min_transp_fill)/(max_transp_fill-min_transp_fill)
   
    #normalizes transp to first value in the fill

    transp = transp/transp[0]
    transp_fill = np.append(transp_fill, transp)
    date_fill = [datetime.datetime.fromtimestamp(ts) for ts in df.time]

    

    #plot transparency in fill k vs absolute time
    plt.figure()
    plt.plot(date_fill, transp, ".b-", markersize=3, linewidth=0.75, label="Observed")
    plt.xlabel("time")
    plt.ylabel("Normalized mean transparency in fill")
    plt.tick_params(labelsize=7)
    plt.title(f"fill num {k}")
    plt.savefig(f'{k}')


    #plt.show()
    a = np.empty(np.size(transp))
    b = np.empty(np.size(transp))
    a.fill(df['lumi_inst'].iloc[0])
    b.fill(df['lumi_int'].iloc[0])
    lumi_inst_0 = np.append(lumi_inst_0, a)
    lumi_int_0 = np.append(lumi_int_0, b)






    