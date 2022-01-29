import sys
import os
import argparse
from itertools import islice
import ROOT as R
import pandas as pd
import numpy as np  
from tqdm import tqdm


f = R.TFile("/home/federico/Scaricati/BlueLaser_2017_rereco_v2_newformat.root")
tree = f.Get("ntu")

pbar = tqdm(total=tree.GetEntries())


outputs = {
    "transp_entry" : [],                        #   counter
    "time": [],                                 #   absolute time of the first laser region in the scan. 
                                                #           FIXME: it could be improved making it xtal dependent, but 20 min window is ok so far. Special case for interrupted sequences will need some attention
    "bfield": [],                               #   magnetic field
    "fill_num": [],                             #   LHC fill number
    "fill_time_start": [],                      #   Time of the beginning of the fill
    "fill_time_stable_start": [],               #   Time of the declaration of stable beam for the current fill
    "time_in_fill": [],                         #   Time since the beginning of the fill
    "time_in_fill_stable": [],                  #   Time since the moment stable beam is declared
    "run_in_fill" : [],                         #   Run number in fill
    "in_fill" : []                              #   Flag: 1 = during a fill, 0 = not during a fill
}


transp_data_EB_tot = []
transp_data_EE_tot = []


if len(sys.argv)>1:
    nevent = int(sys.argv[1])
    if len(sys.argv) > 2:
        nevent2 = int(sys.argv[2])
    else:
        nevent2 = nevent+1
    tree = islice(tree, nevent, nevent2)
 
def add_output(out):
    for k, v in out.items():
        outputs[k].append(v)

last_fill_dump_time = 0


for iev, event in enumerate(tree):
    pbar.update()

    out = {}
    out["time"] = event.time[0]
    out["bfield"] = event.bfield
    out["run_in_fill"] = event.run_num_infill
    out["fill_num"] = event.fill_num
    isinfill = out["fill_num"] != 0
    out["fill_time_start"] = event.fill_time 
    out["fill_time_stable_start"] = event.fill_time_stablebeam
    out["in_fill"] = int(isinfill)

    if isinfill:
        out["time_in_fill"] = event.time[0] - event.fill_time
        time_infill_stable = event.time[0] - event.fill_time_stablebeam
        if time_infill_stable > 0: 
            out["time_in_fill_stable"] = time_infill_stable
        else: 
            out["time_in_fill_stable"] = 0
        
    else:    
        out["time_in_fill"] = 0.
        out["time_in_fill_stable"] = 0.
        

    transp = []

    transp_data_EB = np.zeros((170, 360))
    transp_data_EE = np.zeros((2,100, 100))
    for iz, ix, iy, tr in zip(event.iz, event.ix, event.iy, event.nrv):
        if iz ==0 :
            #barrel
            if ix > 0: ixind = ix + 84
            if ix < 0: ixind = ix + 85

            transp_data_EB[ixind][iy-1] = tr
        else:
            #endcap
            if iz == 1:  izindex = 0
            if iz == -1: izindex = 1
            transp_data_EE[izindex][ix-1][iy-1] = tr

    # print(transp_data_EB)
    # print("=====")
    # print(transp_data_EE)
    transp_data_EB_tot.append(transp_data_EB)
    transp_data_EE_tot.append(transp_data_EE)
    
    out["transp_entry"] = iev

    add_output(out)
    #print(out)

df = pd.DataFrame(outputs)
df.to_csv("transp_metadata_2017_v2.csv", sep=",", index=False)

tdata_EB = np.array(transp_data_EB_tot)
tdata_EE = np.array(transp_data_EE_tot)

np.save("transp_data_EB_v2.npy", tdata_EB)
np.save("transp_data_EE_v2.npy", tdata_EE)
