# @brusale's Thesys 
The aim of the work was to find a proper 2D-functional form for the Transparency of CMS-ECAL xtals (its correlation with Instantaneous luminosity and Integrated luminosity), and evaluate the consequences of its correction on the trigger efficiency.

link to transparency repo: https://github.com/federicodematteis/Transparency

# Data Preparation
link to files: http://dvalsecc.web.cern.ch/dvalsecc/ECAL/Transparency/data_v1/ .

1- LaserDataPreparation.py takes in input the file:
BlueLaser_2017_rereco_v2_newformat.root.
It outputs Transparency datas divided into two npy arrays, and several transparency metadata, with non-regular (non-constant) time-granularity: transp_metadata_2017_v2.csv, transp_data_EE.npy transp_data_EB.npy .

2-TimestepsDataPreparation.py takes the lumi_brilcalc_2017.csv file and outputs several metadata with a 10 min time step: output_metadata_2017_10min.csv (or fill_metadata_2017_10min.csv)

3-output_transp_timestamps.py takes in input:  
transp_data_EE.npy, transp_data_EB.npy, transp_metadata_2017_v2.scv, output_metadata_2017_10min.csv.
It interpolates the value of transparency with non regular and regular time-granularity datas, "generating" a npy array containing values for Transparency, which are now correspondents to the entries in the output_metadata_2017_10min.csv file (or fill_output_2017_10min).

In his work @Brusale used iRing.py to do the same job of output_transp_timestamps.py, extracting values for transparency for all x,y,z in the selected iRing.

# Plotting
We will use only EE datas for now.
Trading the path of @brusale I used the EE_2D_Fitting.py script for evaluating the parameters of fit_func2 on iRing<num>.npy's  mean Transparency entries and to plot several fit-function and bias.
With the same script is possible to test the model on other iRings.

# Expected Performance
Another script is used to convert iRing<num>.npy into .txt files: GetData.py, making them readable for the cxx script).
For evaluating Transparency correction's effect on the trigger efficiency @Brusale used TurnOnCurve.cxx which fills three different hist: -efficiency real, efficiency corrected and Ideal efficiency (the step function).
 
The main focus was to have a starting point: TurnOnCurve for real Transparency datas, and a target: the step function, using parameters found with EE_2D_Fitting on some iRings

# Machine Learning 
The aim of the current work : Transparency predictions (for a single fill) with a DNN regression.
 
(thanks to @brusale, @amassiro, @valsdav)
