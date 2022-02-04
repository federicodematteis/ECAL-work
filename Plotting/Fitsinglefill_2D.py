from re import X
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from pprint import pprint
from collections import namedtuple
import datetime
import matplotlib.colors as colors
from scipy.optimize import curve_fit
from scipy.stats import chisquare

mpl.rcParams['figure.figsize'] = (5,4)
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams["image.origin"] = 'lower'
 
#get_ipython().run_line_magic('matplotlib', 'notebook')

data_folder = ('/home/federico/root/root-6.24.06-install')
metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")
#load data for i ring 1 
data = np.load(f"{data_folder}/iRing26new.npy")
data_df = pd.DataFrame(data)

#Mean transparency in iRing
mean = []

for i in range(0, len(data_df.axes[1])):
    mean = np.append(mean, np.mean(data_df[i]))

mean = mean[mean != -1]
metadata = metadata.iloc[:len(mean)][mean != -1]

##---- fit 2D ----
#definisco la funzione 2d:
def fit_func_2(datas, a, b, c, d, e, f):
    x = datas[0]
    y = datas[1]
    y_0 = transp_6371[0]*(1e-9)#8.004434*(1e-9)
    return (a*np.exp(-b*x)+(1-a)*np.exp(c*x))*(d*np.exp(-e*(y-y_0))+(1-d)*np.exp(f*(y-y_0)))


##------FILL 6371------##

# metadata for fill 6371
metadata_6371 = metadata[metadata.fill_num == 6371]
#filter lumi_in_fill e lumi_inst
metadata_6371 = metadata_6371[(metadata_6371.lumi_inst >= 0.0001*1e9) & (metadata_6371.lumi_inst <= 0.0004*1e9) & (metadata_6371.lumi_in_fill >= 0.1*1e9)]
transp_6371 = mean[metadata_6371.index.values[0]:metadata_6371.index.values[0]+len(metadata_6371.axes[0])]
# normalizzo il dato di trasparenza a quella precedente al fill
transp_6371 = transp_6371/transp_6371[0]

#fit parameters found by fitting iRings - fill 6371
par25 = [0.09192309,2.85013885,0.15955624,9.20390233,-3.23549497,0.99469655]
par26 = [0.0912546,3.8466344,0.15864503,17.50818141,-3.03488584,0.84651477]
par24 = [0.08717691,3.71098111,0.14912243,15.01841404,-3.06543558,0.73526334]
par23 = [0.08334285,3.73442396,0.14184352,13.85424341,-3.34668462,0.91228228]

#---------- PLOT trasparenza fittata con la forma di doppio esponenziale 2D -----------

plt.plot(metadata_6371.time, fit_func_2([metadata_6371.lumi_in_fill*(1e-9), metadata_6371.lumi_inst*(1e-9)], *par25), ".c-", markersize=2, linewidth=0.75, label="fill 6371, prediction from iring 25")
plt.plot(metadata_6371.time, fit_func_2([metadata_6371.lumi_in_fill*(1e-9), metadata_6371.lumi_inst*(1e-9)], *par26), ".m-", markersize=2, linewidth=0.75, label="fill 6371, predicted from iring 26")
plt.plot(metadata_6371.time, fit_func_2([metadata_6371.lumi_in_fill*(1e-9), metadata_6371.lumi_inst*(1e-9)], *par24), ".b-", markersize=2, linewidth=0.75, label="fill 6371, predicted from iring 24")
plt.plot(metadata_6371.time, fit_func_2([metadata_6371.lumi_in_fill*(1e-9), metadata_6371.lumi_inst*(1e-9)], *par23), ".g-", markersize=2, linewidth=0.75, label="fill 6371, predicted from iring 23")

plt.plot(metadata_6371.time, transp_6371, ".r-", markersize=2, linewidth=0.75, label='measured fill 6371')
plt.legend(title='prediction on iRing 26')
plt.xlabel('time')
plt.ylabel('Predicted & Measured mean transparency')
plt.tick_params(labelsize=7)
plt.title('Transparency vs time - single fill')
#plt.savefig('fit_exp_exp')
plt.show()

##-------FILL 5962-------

# metadata for fill 6371
metadata_5962 = metadata[metadata.fill_num == 5962]
#filter lumi_in_fill e lumi_inst
metadata_5962 = metadata_5962[(metadata_5962.lumi_inst >= 0.0001*1e9) & (metadata_5962.lumi_inst <= 0.0004*1e9) & (metadata_5962.lumi_in_fill >= 0.1*1e9)]
transp_5962 = mean[metadata_5962.index.values[0]:metadata_5962.index.values[0]+len(metadata_5962.axes[0])]
# normalizzo il dato di trasparenza a quella precedente al fill
transp_5962 = transp_5962/transp_5962[0]


#fit parameters found by fitting iRings - fill 5962
par26 = [0.0439635,9.52461282,0.06050369,23.55811403,-4.03191959,0.97508831]
par25 = [9.99999958e-01,2.38597330e-02,3.63776541e+01,-2.36672498e-01,1.25178511e+01,1.00519322e+00]
par24 = [0.03909184,9.11703406,0.05344768,22.0592916,-3.77427049,0.98365022]
par23 = [0.037259,8.72289079,0.05183301,14.48198252,-4.85845398,0.97573511]

#---------- PLOT trasparenza fittata con la forma di doppio esponenziale 2D -----------

plt.plot(metadata_5962.time, fit_func_2([metadata_5962.lumi_in_fill*(1e-9), metadata_5962.lumi_inst*(1e-9)], *par25), ".c-", markersize=2, linewidth=0.75, label="fill 5962, prediction from iring 25")
plt.plot(metadata_5962.time, fit_func_2([metadata_5962.lumi_in_fill*(1e-9), metadata_5962.lumi_inst*(1e-9)], *par26), ".m-", markersize=2, linewidth=0.75, label="fill 5962, predicted from iring 26")
plt.plot(metadata_5962.time, fit_func_2([metadata_5962.lumi_in_fill*(1e-9), metadata_5962.lumi_inst*(1e-9)], *par24), ".b-", markersize=2, linewidth=0.75, label="fill 5962, predicted from iring 24")
plt.plot(metadata_5962.time, fit_func_2([metadata_5962.lumi_in_fill*(1e-9), metadata_5962.lumi_inst*(1e-9)], *par23), ".g-", markersize=2, linewidth=0.75, label="fill 5962, predicted from iring 23")

plt.plot(metadata_5962.time, transp_5962, ".r-", markersize=2, linewidth=0.75, label='measured fill 5962')
plt.legend(title='prediction on iRing 26')
plt.xlabel('time')
plt.ylabel('Predicted & Measured mean transparency')
plt.tick_params(labelsize=7)
plt.title('Transparency vs time - single fill')
#plt.savefig('fit_exp_exp')
plt.show()

#-------- FIT IRING 6180 -------

# metadata for fill 6371
metadata_6297 = metadata[metadata.fill_num == 6297]
#filter lumi_in_fill e lumi_inst
metadata_6297 = metadata_6297[(metadata_6297.lumi_inst >= 0.0001*1e9) & (metadata_6297.lumi_inst <= 0.0004*1e9) & (metadata_6297.lumi_in_fill >= 0.1*1e9)]
transp_6297 = mean[metadata_6297.index.values[0]:metadata_6297.index.values[0]+len(metadata_6297.axes[0])]
# normalizzo il dato di trasparenza a quella precedente al fill
transp_6297 = transp_6297/transp_6297[0]

par23=[0.10722318,3.40142748,0.16528638,3.44148397,-14.4458373,1.00400629]
par24=[0.10979645,3.46056918,0.17069329,-9.31489117,4.18862643,1.04023372]
par25=[0.10798524,2.71326079,0.167472,8.46255498,-4.18165624,0.99664376]
par26=[0.10798524,2.71326079,0.167472,8.46255498,-4.18165624,0.99664376]

#----------PLOT trasparenza fittata con la forma di doppio esponenziale 2D----------
plt.plot(metadata_6297.time, fit_func_2([metadata_6297.lumi_in_fill*(1e-9), metadata_6297.lumi_inst*(1e-9)], *par25), ".c-", markersize=2, linewidth=0.75, label="fill 6297, prediction from iring 25")
plt.plot(metadata_6297.time, fit_func_2([metadata_6297.lumi_in_fill*(1e-9), metadata_6297.lumi_inst*(1e-9)], *par26), ".m-", markersize=2, linewidth=0.75, label="fill 6297, predicted from iring 26")
plt.plot(metadata_6297.time, fit_func_2([metadata_6297.lumi_in_fill*(1e-9), metadata_6297.lumi_inst*(1e-9)], *par24), ".b-", markersize=2, linewidth=0.75, label="fill 6297, predicted from iring 24")
plt.plot(metadata_6297.time, fit_func_2([metadata_6297.lumi_in_fill*(1e-9), metadata_6297.lumi_inst*(1e-9)], *par23), ".g-", markersize=2, linewidth=0.75, label="fill 6297, predicted from iring 23")

plt.plot(metadata_6297.time, transp_6297, ".r-", markersize=2, linewidth=0.75, label='measured fill 6297')
plt.legend(title='prediction on iRing 26')
plt.xlabel('time')
plt.ylabel('Predicted & Measured mean transparency')
plt.tick_params(labelsize=7)
plt.title('Transparency vs time - single fill')
#plt.savefig('fit fill 5962)
plt.show()

#notiamo che la predizione per gli i ring 24, 25, 26 sono molto simili tra loro, mentre la predizione 
#per i Ring 25 si discosta dalle altre, iRing25 non è un ottimo candidato per fare predizione sullo stesso fill di altri iRing.
#in realà pe rottenere la predizione su altri i Ring(diversi da 26) basta cambiare il file numpy in input alla riga 22


