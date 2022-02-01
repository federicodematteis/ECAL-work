#using file: Plot_mean_transparency_iRing-checkpoint.ipynb
import matplotlib.pyplot as plt
import ROOT
import matplotlib as mpl
import pandas as pd
import numpy as np
from pprint import pprint
from collections import namedtuple
import datetime

mpl.rcParams['figure.figsize'] = (5,4)
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams["image.origin"] = 'lower'
# In[ ]:
data_folder = ('/home/federico/root/root-6.24.06-install')
data1 = np.load(f"{data_folder}/iRing25new.npy")
data1_df = pd.DataFrame(data1)
data1_df.head()
mean1 = []
# In[ ]:
for i in range (0, len(data1_df.axes[1])):
    #mean of colums' entries of data_df (for each "position" in the cristal)
    mean1 = np.append(mean1, np.mean(data1_df[i]))
    
mean1= mean1[mean1 != -1]

#read metadata file with same t-granularity of iRing25new.npy's T. values 
metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")
metadata = metadata.iloc[:len(mean1)][mean1!=-1]
date = [datetime.datetime.fromtimestamp(ts) for ts in metadata.time]
# In[ ]:
#plot mean tarnsparency in iRing 25
plt.plot(date, mean1, ".b-", markersize=1, linewidth=0.5, label='iRing = 25 ')
plt.legend(loc='lower left')
plt.tick_params(labelsize=7)
plt.xticks(rotation='45')
plt.ylabel('Mean transparency real - entire run')
plt.show()

# In[ ]:
metadata.head()
#---------------------------------------------------
#Restrict dataframe to fill 6371
metadata1 = metadata[metadata['fill_num'] == 6371] 
date1 = [datetime.datetime.fromtimestamp(ts) for ts in metadata1.time]

metadata1.head()
mean_1 = mean1[metadata1.index.values[0]:metadata1.index.values[0]+len(metadata1.axes[0])]
#print(np.size(mean_1))

# plt.plot(date1, mean_1, ".r-", markersize=2, linewidth=0.75, label='iRing 25, fill 6371')
# plt.xticks(rotation ='45')
# plt.tick_params(labelsize=5)
# plt.legend()
# plt.ylabel('Mean transparency in fill')
# plt.show()

# In[ ]:
#This is the function used in TurnOnCurve.cxx to correct transparency
def fit_func2(data, a, b, c, d, e, f):
    x = data[0]
    y = data[1]
    y_0 = data[2]
    return (a*np.exp(-b*x)+(1-a)*np.exp(c*x))*(d*np.exp(-e*(y-y_0))+(1-d)*np.exp(f*(y-y_0)))

par2 = [0.99327073,0.03867906,3.22509689,7.48668825,2.61653155,-2.93094313]

print(metadata)
#inserire non a mano 
lumi_inst_0 = 8.004434

#mean transparency in iRing corrected vs real - signle fill
plt.plot(date1, (mean_1/mean1[metadata1.index.values[0]-1])/fit_func2([metadata1.lumi_in_fill*(1e-9), metadata1.lumi_inst*(1e-9), lumi_inst_0*(1e-9)], *par2), ".r-", markersize=2, linewidth=0.75, label="iRing 25 corrected, fill 6371" )
plt.plot(date1, mean_1/(mean1[metadata1.index.values[0]-1]), ".b-", markersize=2, linewidth=0.75, label="iRing 25, fill 6371")
plt.xticks(rotation='45')
plt.tick_params(labelsize=5)
plt.legend()
plt.ylabel('Normalized mean transparency before and after correction - fill 3671')
plt.show()

# In[ ]:
# mean transparency in iRing corrected vs real - entire run
plt.plot(date, (mean1)/fit_func2([metadata.lumi_in_fill*(1e-9), metadata.lumi_inst*(1e-9), lumi_inst_0*(1e-9)], *par2), ".r-", markersize=1, linewidth=0.5, label="iRing 25 corrected, fill 6371" )
plt.xticks(rotation='45')
plt.tick_params(labelsize=5)
plt.legend()
plt.ylabel('Mean transparency after correction - entire run')
plt.show()

# codice jupyter e caricare questi plot e quelli T vs lumi_in_fill-lumi_inst


