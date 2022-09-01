#using file: Plot_mean_transparency_iRing-checkpoint.ipynb
import matplotlib.pyplot as plt
import ROOT
import matplotlib as mpl
import pandas as pd
import numpy as np
from pprint import pprint
from collections import namedtuple
import datetime
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from sklearn.metrics import mean_squared_error

mpl.rcParams['figure.figsize'] = (5,4)
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams["image.origin"] = 'lower'
# In[ ]:
data_folder = ('/home/federico/root/root-6.24.06-install')
data1 = np.load(f"{data_folder}/iRing25new.npy")
data1_df = pd.DataFrame(data1)
data1_df.head()
mean0 = []
# In[ ]:
for i in range (0, len(data1_df.axes[1])):
    #mean of colums' entries of data_df (for each "position" in the cristal)
    mean0 = np.append(mean0, np.mean(data1_df[i]))
    
mean0= mean0[mean0 != -1]

#read metadata file with same t-granularity of iRing25new.npy's T. values 
metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")
metadata = metadata.iloc[:len(mean0)][mean0!=-1]
date = [datetime.datetime.fromtimestamp(ts) for ts in metadata.time]

# In[ ]:
#plot mean tarnsparency in iRing 25
plt.plot(date, mean0, ".b-", markersize=1, linewidth=0.5, label='iRing = 25 ')
plt.legend(loc='lower left')
plt.tick_params(labelsize=7)
plt.xticks(rotation='45')
plt.ylabel('Mean transparency in iRing')
plt.show()


#---------------------------------------------------
#Restrict dataframe to fill 6371

metadata0 = metadata[metadata['fill_num'] == 6371] 
metadata0 = metadata0[(metadata0.lumi_inst >= 0.0001*1e9) & (metadata0.lumi_inst <= 0.0004*1e9) & (metadata0.lumi_in_fill >= 0.1*1e9)]

print(metadata0)

date0 = [datetime.datetime.fromtimestamp(ts) for ts in metadata0.time]

#metadata1.head()
mean_0 = mean0[metadata0.index.values[0]:metadata0.index.values[0]+len(metadata0.axes[0])]
#print(np.size(mean_1))

#----Plot transparency during the fill
# plt.plot(date0, mean_0, ".r-", markersize=2, linewidth=0.75, label='iRing 25, fill 6371')
# plt.xticks(rotation ='45')
# plt.tick_params(labelsize=5)
# plt.legend()
# plt.ylabel('Mean transparency')
#plt.show()

#----Plot normalized mean transparency
#Trasparenza normalizzata alla trasparenza appena precedente al fill
plt.plot(date0, mean_0/(mean0[metadata0.index.values[0]-1]), ".r-", markersize=2, linewidth=0.75, label="iRing 0, fill 6371")
plt.xticks(rotation='45')
plt.tick_params(labelsize=5)
plt.legend()
plt.ylabel('Normalized mean transparency')
plt.show()

transp_fill = []

df = metadata0[metadata0.fill_num == 6371]

print(df.index.values)
print("trasparenza media" )
print((mean0))

transp = [mean0[i] for i in df.index.values]
transp = transp/transp[0]
transp_fill = np.append(transp_fill, transp)


print("transp_fill 6371")
print(transp_fill)
#è cio che uso in curve fitting
print("metadata for curve fitting")
print(metadata0)

#This is the function used in TurnOnCurve.cxx to correct transparency
def fit_func2(data, a, b, c, d, e, f):
    x = data[0]
    y = data[1]
    #y_0 = data[2]
    y_0 = 8.004434*(1e-9)
    return (a*np.exp(-b*x)+(1-a)*np.exp(c*x))*(d*np.exp(-e*(y-y_0))+(1-d)*np.exp(f*(y-y_0)))

def fit_func1(data, a, b, c, d, e, f):
    x = data[0]
    y = data[1]
    y_0 = data[2]
    return (a*np.exp(-b*x)+(1-a)*np.exp(c*x))*(d*np.exp(-e*y)+(1-d)*np.exp(f*y))/(d*np.exp(-e*y_0)+(1-d)*np.exp(f*y_0))


lumi_inst_0 = 8.004434 
#par1 =[9.93269836e-01,3.86809394e-02,3.22496804e+00,-7.94580199e+04,2.13923206e+05,-5.73455919e-01]
par2=[0.99327073,0.03867906,3.22509689,7.48668825,2.61653155,-2.93094313] #iring25
#par3=[0.99577206,0.0407842,3.71421979,6.29341453,1.80420883,-3.01453359]#iring24
#par4=[9.93270514e-01,3.86793551e-02,3.22506673e+00,-7.34856114e+04,1.98245082e+05,-5.75097262e-01]


plt.plot(metadata0.time, transp_fill, ".b-", markersize=2, linewidth=0.75, label="iRing 25, fill 6371, measured")
plt.plot(metadata0.time, fit_func2([metadata0.lumi_in_fill*(1e-9), metadata0.lumi_inst*(1e-9), lumi_inst_0*(1e-9)], *par2), ".r-", markersize=2, linewidth=0.75, label="iRing 25, fill 6371, predicted")
plt.xticks(rotation='45')
plt.tick_params(labelsize=5)
plt.legend()
plt.ylabel('Normalized mean transparency')
plt.show()


print([metadata0.lumi_in_fill*(1e-9), metadata0.lumi_inst*(1e-9)])
#transp fill dovrebbe essere la trasparenza media nel fill selezionato in questo script: 
#in EE2Dfitting è 
par2, pcov2 = curve_fit(fit_func2, [metadata0.lumi_in_fill*(1e-9), metadata0.lumi_inst*(1e-9)], transp_fill, maxfev=5000)
print(par2)
# #@brusale : par2 = [0.993,3.87e-2,3.22,10,2.71,-3.02]







# print(metadata1)
# #inserire non a mano 
# lumi_inst_0 = 8.004434

#posso eseguire il fit direttamente qui ?
#forse sarebbe meglio, prendo transp di un particolare fill e faccio l fit
#tanto la funzione è gia defiita e il valore per esempio di transp nel fill 
#6371 è salvato in mean_0; così faccio un fit alla volta e poi plotto 
#quella intera (cioè per tutt i fill) con un for ad esempio

# #mean transparency in iRing predicted vs real - signle fill
# plt.plot(date1, fit_func2([metadata1.lumi_in_fill*(1e-9), metadata1.lumi_inst*(1e-9), lumi_inst_0*(1e-9)], *par2), ".r-", markersize=2, linewidth=0.75, label="iRing 25 predicted, fill 6371" )
# plt.plot(metadata1.time, mean_1/(mean1[metadata1.index.values[0]-1]), ".b-", markersize=2, linewidth=0.75, label="iRing 25, fill 6371")
# plt.xticks(rotation='45')
# plt.tick_params(labelsize=5)
# plt.legend()
# plt.ylabel('Normalized mean transparency predicted - fill 3671')
# plt.show()

# # In[ ]:
# # mean transparency in iRing predicted vs real - entire run
# plt.plot(date, fit_func2([metadata.lumi_in_fill*(1e-9), metadata.lumi_inst*(1e-9), lumi_inst_0*(1e-9)], *par2), ".r-", markersize=1, linewidth=0.5, label="iRing 25 entire run" )
# plt.xticks(rotation='45')
# plt.tick_params(labelsize=5)
# plt.legend()
# plt.ylabel('Mean transparency predicted - entire run')
# plt.show()




