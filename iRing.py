#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from pprint import pprint
from collections import namedtuple
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--transparency-metadata", help="Transparency metadata file", type=str)
parser.add_argument("--timestamps-metadata", help="Timestamps metadata file", type=str)
parser.add_argument("-o","--output-file", help="Output file", type=str)
parser.add_argument("--iring", help="iring", type=int)
parser.add_argument("-d","--data", help="Data folder")
args = parser.parse_args()


# In[ ]:


data_EE = np.load(f"{args.data}/transp_data_EE_v2.npy", mmap_mode="r")
data_EB = np.load(f"{args.data}/transp_data_EB_v2.npy", mmap_mode="r")


# ## Read brilcalc metadata

meta = pd.read_csv(args.transparency_metadata,sep=",").astype(
            {"transp_entry":"int64", "time": "int64", "fill_num":"int64", "time_in_fill":"int64", "time_in_fill_stable":"int64"})


timestamps_df = pd.read_csv(args.timestamps_metadata, sep=",", comment="#")


# In[ ]:


iring_df = pd.read_csv('https://raw.githubusercontent.com/amassiro/EcalPF/master/test/draw/eerings.dat', sep='\s', header=None, names=['ix', 'iy', 'iz', 'iring'], engine='python')


# In[ ]:

#restringe la lista a quella corrispondente a iring 1 (dato come argument) è un panda dataframe
iring_df = iring_df[iring_df['iring'] == args.iring]


# In[ ]:
#definisce le funzioni con cui fa get 

def get_transp_EB(index, x, y):
    EB = data_EB[int(index)-1][x, y]
    return EB

def get_transp_EE(index, z, x, y):
    EE = data_EE[int(index)-1][z][x, y]
    return EE

def get_transp_interval(timestamp):
    last_meas = meta[(meta.time < timestamp)].iloc[-1]
    next_meas = meta[(meta.time > timestamp)]
    if next_meas.empty:
        return last_meas, pd.DataFrame()
    #print("{} {} | {}| x0: {} | Interval diff {:.3f}".format(last_meas.time, next_meas.time,  timestamp,timestamp- last_meas.time, ( next_meas.time - last_meas.time) / 60))
    return last_meas, next_meas.iloc[0]

#definisce le funzioni con cui intepola :

#Trasparenza

def interpolate_transp(x, x0, x1, y0, y1):
    z = (x - x0)*( (y1-y0)/(x1-x0)) + y0
    #print(f"x {x}, x0 {x0}, x1 {x1}, y0 {y0}, y1 {y1} ---> {z}")
    return z


def get_transp_interpolate(timestamp, x, y, z):
    Z0, Z1 = get_transp_interval(timestamp)
    if Z1.empty:
        return -1.

    if z == 0:
        transp_EB_y0 = get_transp_EB(Z0.transp_entry, x, y)
        transp_EB_y1 = get_transp_EB(Z1.transp_entry, x, y)
        trans_EB = interpolate_transp(timestamp, Z0.time, Z1.time,  transp_EB_y0, transp_EB_y1)
        return trans_EB
    
    if z == 1:
        transp_EE_y0 = get_transp_EE(Z0.transp_entry, 0, x, y)
        transp_EE_y1 = get_transp_EE(Z1.transp_entry, 0, x, y)
        trans_EE =  interpolate_transp(timestamp,  Z0.time, Z1.time,  transp_EE_y0, transp_EE_y1)
        return trans_EE

    if z == -1:
        transp_EE_y0 = get_transp_EE(Z0.transp_entry, 1, x, y)
        transp_EE_y1 = get_transp_EE(Z1.transp_entry, 1, x, y)
        trans_EE =  interpolate_transp(timestamp,  Z0.time, Z1.time,  transp_EE_y0, transp_EE_y1)
        return trans_EE
   
    


# In[ ]:

#dichiara transp_output
transp_output = []
#dichiara tot
tot = len(timestamps_df)
    
#main for: fa uno scan sui tempi dentro uno scan su len(iring_df.axes[0]) cioè sulla lunghezza delle righe di iring_df (quindi sulla sua dimensione!), quindi per ogni riga
for i in range(0, len(iring_df.axes[0])):
    print(f"{i}/{len(iring_df.axes[0])}")
    #definisce tran per usarlo nel for sui tempi
    tran = []
    #scan sui tempi (per ogni riga) e per ogni t (quindi per tutto il tempo a disposizione in round da 10 minuti???)
    for iev, row in timestamps_df.iterrows():
        #if iev % 100 == 0:
            #print(f"{iev}/{tot}")
            #aggiunge una riga a tran riempiendola con il valore di T interpolata 
            #T interpolata è calcolata in get_transp_interpolate con argomenti (time,iring_df.iloc[][])-1 
            tran.append(get_transp_interpolate(row.time, (iring_df.iloc[i]['ix'])-1, (iring_df.iloc[i]['iy'])-1, iring_df.iloc[i]['iz']))
    
    transp_output.append(tran)
    
np.save(args.output_file,  np.array(transp_output))


# In[ ]:
