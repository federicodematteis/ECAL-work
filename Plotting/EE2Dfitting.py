#Using EE_2d_Fitting.py

#We want the fit-parameters for function 2 for the fit-iRing (function needed for the transparency correction )
#Let's use @brusale's script to get parameters for the fit-iring, changing iRing.npy files in rows 19-20

#We will have two series of parameters as output on terminal.
#We are interested in the second one, we will copy that in the TunOnCurve.cxx script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from sklearn.metrics import mean_squared_error

data_folder=("/home/federico/root/root-6.24.06-install")

#-- Load data and metadata --

metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")
data = np.load(f"{data_folder}/iRing25new.npy")
data_test_2 = np.load(f"{data_folder}/iRing24.npy")

#-- Dataframes --

data_df = pd.DataFrame(data)
#print(data_df)
data_df_test_2 = pd.DataFrame(data_test_2)

#-- mean transparency in iRing --

mean = []          #Fit-iRing
mean_test_2 = []   #Test-iRing

for i in range(0, len(data_df.axes[1])):

    mean = np.append(mean, np.mean(data_df[i]))  
    #if i == 10 :
       # print(data_df[i])
                            
    mean_test_2 = np.append(mean_test_2, np.mean(data_df_test_2[i]))   

#-- Filter data and metadata --

mean = mean[mean != -1]
mean_test_2 = mean_test_2[mean_test_2 != -1]
metadata = metadata.iloc[:len(mean)][mean != -1]
#print("nuova size di mean")
#print(np.size(mean))
#print("metadata")
#print(metadata)
#print(np.size(metadata))

#in mean sono salvati i dati di trasparenza reale pr tutto il run
#infatti ha la stessa size di mean1 in TiRing1.py

#-- Defining fit functions --

def fit_func1(data, a, b, c, d, e, f):
    x = data[0]
    y = data[1]
    y_0 = data[2]
    return (a*np.exp(-b*x)+(1-a)*np.exp(c*x))*(d*np.exp(-e*y)+(1-d)*np.exp(f*y))/(d*np.exp(-e*y_0)+(1-d)*np.exp(f*y_0))

#This is the function used in TurnOnCurve.cxx to fill Trigger efficiency hist
def fit_func2(data, a, b, c, d, e, f):
    x = data[0]
    y = data[1]
    y_0 = data[2]
    return (a*np.exp(-b*x)+(1-a)*np.exp(c*x))*(d*np.exp(-e*(y-y_0))+(1-d)*np.exp(f*(y-y_0)))

#-----------------------------------------

fill = metadata["fill_num"].unique()
#print(len(fill))
#toglie l'elemento iniziale a fill 
fill = fill[fill != 0]
#print(len(fill))
#restringe metadata a quello enza il fill 0
metadata_fill = metadata[metadata.fill_num.isin(fill)]
metadata_fill = metadata_fill[(metadata_fill.lumi_inst >= 0.0001*1e9) & (metadata_fill.lumi_inst <= 0.0004*1e9) & (metadata_fill.lumi_in_fill >= 0.1*1e9)]

fill_num = metadata_fill.fill_num.unique()

#print(fill_num)



# getting fit-iRing datas and metadatas: lumi_inst, lumi_int, transparency

transp_fill = []
lumi_inst_0 = []
lumi_int_0 = []


for k in fill_num:
    #restringe metadata a quello di un solo fill (k)
    #viene fatto per ogni fill
    df = metadata_fill[metadata_fill.fill_num == k]
    transp = [mean[i] for i in df.index.values]
    #transp ha la grandezza del dataframe ristretto al k esimo fill
    transp = transp/transp[0]
    transp_fill = np.append(transp_fill, transp)
    a = np.empty(np.size(transp))
    b = np.empty(np.size(transp))
    a.fill(df['lumi_inst'].iloc[0])
    b.fill(df['lumi_int'].iloc[0])
    lumi_inst_0 = np.append(lumi_inst_0, a)
    lumi_int_0 = np.append(lumi_int_0, b)
    


plt.scatter(metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), c=transp_fill, cmap='jet', s=3)
plt.ylim(0, 0.0005)
cbar = plt.colorbar()
plt.tick_params(labelsize=6)
plt.xlabel("Lumi in fll [fb$^{-1}$]")
plt.ylabel("Instantaneous luminosity [fb$^{-1}$/s]")
cbar.set_label("Normalized transparency")
plt.savefig("fill")
plt.show()

#---Fit Function 1-----------

par1, pcov1 = curve_fit(fit_func1, [metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), lumi_inst_0*(1e-9)], transp_fill, maxfev=5000)
#print(par1)

#Plot fit function #1
plt.scatter(metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), c=fit_func1([metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9),lumi_inst_0*(1e-9)], *par1), cmap='jet', s=3)
plt.ylim(0, 0.0005)
cbar1 = plt.colorbar()
plt.tick_params(labelsize=6)
plt.xlabel("Lumi in fll [fb$^{-1}$]")
plt.ylabel("Instantaneous luminosity [fb$^{-1}$]")
cbar1.set_label("Fit function")
#plt.savefig("fill_fit1")
plt.show()

#Plot transp_fill-fit_function #1
plt.scatter(metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), c=transp_fill-fit_func1([metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9),lumi_inst_0*(1e-9)], *par1), cmap='jet', s=3)
plt.ylim(0, 0.0005)
cbar2 = plt.colorbar()
plt.tick_params(labelsize=6)
plt.xlabel("Lumi in fll [fb$^{-1}$]")
plt.ylabel("Instantaneous luminosity [fb$^{-1}$]")
cbar2.set_label("Observed-Predicted")
#plt.savefig("fill_fit1_bias")
plt.show()

#Plot (transp_fill-fit_function #1)/transp-fill
plt.scatter(metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), c=(transp_fill-fit_func1([metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9),lumi_inst_0*(1e-9)], *par1))/transp_fill, cmap='jet', s=3)
plt.ylim(0, 0.0005)
cbar3 = plt.colorbar()
plt.tick_params(labelsize=6)
plt.xlabel("Lumi in fll [fb$^{-1}$]")
plt.ylabel("Instantaneous luminosity [fb$^{-1}$]")
cbar3.set_label("(Observed-Predicted)/Observed")
#plt.savefig("fill_fit1_bias_normalized")
plt.show()

#Plot vs lumi_in_fill
plt.plot(metadata_fill.lumi_in_fill*(1e-9), transp_fill,".r", label="Observed")
plt.plot(metadata_fill.lumi_in_fill*(1e-9), fit_func1([metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), lumi_inst_0*(1e-9)], *par1), ".b", label="Predicted")
plt.xlabel("Lumi in fill [fb$^{-1}$]")
plt.ylabel("Normalized mean transparency")
plt.title("iRing 25")
#plt.savefig("fit1_vs_lumi_in_fill")
plt.show()


#----------- FIT FUNCTION 2 ---------

#fit-iRing

par2, pcov2 = curve_fit(fit_func2, [metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), lumi_inst_0*(1e-9)], transp_fill, maxfev=5000)
print(par2)
# print(len(fit_func2([metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), lumi_inst_0*(1e-9)], *par2)))
# print(len(metadata_fill.lumi_in_fill))

#Plot fit function #2
plt.scatter(metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), c=fit_func2([metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), lumi_inst_0*(1e-9)], *par2), cmap='jet', s=3)
plt.ylim(0, 0.0005)
cbar4 = plt.colorbar()
plt.tick_params(labelsize=6)
plt.xlabel("Lumi in fll [fb$^{-1}$]")
plt.ylabel("Instantaneous luminosity [fb$^{-1}$]")
cbar4.set_label("Fit function")
plt.savefig("fill_fit2")
plt.show()

#Plot transp_fill-fit_function #2
plt.scatter(metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), c=transp_fill-fit_func2([metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9),lumi_inst_0*(1e-9)], *par2), cmap='jet', s=3)
plt.ylim(0, 0.0005)
cbar5 = plt.colorbar()
plt.tick_params(labelsize=6)
plt.xlabel("Lumi in fll [fb$^{-1}$]")
plt.ylabel("Instantaneous luminosity [fb$^{-1}$]")
cbar5.set_label("Observed-Predicted")
plt.savefig("fill_fit2_bias")
plt.show()

#Plot (transp_fill-fit_function #2)/transp-fill
plt.scatter(metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), c=(transp_fill-fit_func2([metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9),lumi_inst_0*(1e-9)], *par2))/transp_fill, cmap='jet', s=3)
plt.ylim(0, 0.0005)
cbar6 = plt.colorbar()
plt.tick_params(labelsize=6)
plt.xlabel("Lumi in fll [fb$^{-1}$]")
plt.ylabel("Instantaneous luminosity [fb$^{-1}$]")
cbar6.set_label("(Observed-Predicted)/Observed")
plt.savefig("fill_fit2_bias_normalized")
plt.show()

#Plot vs lumi_in_fill
plt.plot(metadata_fill.lumi_in_fill*(1e-9), transp_fill,".r", label="Observed")
plt.plot(metadata_fill.lumi_in_fill*(1e-9), fit_func2([metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), lumi_inst_0*(1e-9)], *par2), ".b", label="Predicted")
plt.xlabel("Lumi in fill [fb$^{-1}$]")
plt.ylabel("Normalized mean transparency")
plt.title("iRing 25")
#plt.savefig("fit2_vs_lumi_in_fill")
plt.show

#----TEST on IRING 24 -------

#riempie transp_fill con la trasparenza di iRing25 per ogni fill
transp_fill = []
for k in fill_num:
    df = metadata_fill[metadata_fill.fill_num == k]
    transp = [mean_test_2[i] for i in df.index.values]
    transp = transp/transp[0]
    transp_fill = np.append(transp_fill, transp)

#transp_fill contiene la trasparenza per ogni fill, ha la dimensione del numero di fill 
#print(transp_fill)

# Plot fit function #1
plt.scatter(metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), c=transp_fill-fit_func1([metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9),lumi_inst_0*(1e-9)], *par1), cmap='jet', s=3)
plt.ylim(0, 0.0005)
cbar7 = plt.colorbar()
plt.tick_params(labelsize=6)
plt.xlabel("Lumi in fll [fb$^{-1}$]")
plt.ylabel("Instantaneous luminosity [fb$^{-1}$]")
cbar7.set_label("Observed-Predicted")
#plt.savefig("fill_fit1_test_bias")

# Plot fit function #1
plt.scatter(metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), c=(transp_fill-fit_func1([metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9),lumi_inst_0*(1e-9)], *par1))/transp_fill, cmap='jet', s=3)
plt.ylim(0, 0.0005)
cbar8 = plt.colorbar()
plt.tick_params(labelsize=6)
plt.xlabel("Lumi in fll [fb$^{-1}$]")
plt.ylabel("Instantaneous luminosity [fb$^{-1}$]")
cbar8.set_label("(Observed-Predicted)/Observed")
#plt.savefig("fill_fit1_test_bias_normalized")

# Plot fit function #2
plt.scatter(metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), c=transp_fill-fit_func2([metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9),lumi_inst_0*(1e-9)], *par2), cmap='jet', s=3)
plt.ylim(0, 0.0005)
cbar9 = plt.colorbar()
plt.tick_params(labelsize=6)
plt.xlabel("Lumi in fll [fb$^{-1}$]")
plt.ylabel("Instantaneous luminosity [fb$^{-1}$]")
cbar9.set_label("Observed-Predicted")
#plt.savefig("fill_fit2_test_bias")

#-------------fitting iRing26---------------

# Plot fit function #2
plt.scatter(metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), c=(transp_fill-fit_func2([metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9),lumi_inst_0*(1e-9)], *par2))/transp_fill, cmap='jet', s=3)
plt.ylim(0, 0.0005)
cbar10 = plt.colorbar()
plt.tick_params(labelsize=6)
plt.xlabel("Lumi in fill [fb$^{-1}$]")
plt.ylabel("Instantaneous luminosity [fb$^{-1}$]")
cbar10.set_label("(Observed-Predicted)/Observed")
#plt.savefig("fill_fit2_test_bias_normalized")






