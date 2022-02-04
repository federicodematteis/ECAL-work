## plots
Plots named Transp_prediction_entire_run_num.png are plot of Transparency vs time predicted and measured on the entire "run time".
These plots were obtained using the script EE_2D_Fitting : it fits the fit_func_2 with transparency data in a selected iRing (23-24-25-26),
then uses the fit-parameters for plotting the fit_func_2 as a prediction for Transparency.
We see the behaviour over time of predicted values for Transparency in red, and measured values in blue.
  
Plots named predictionfillnum.png are the plots for Transparency predicted and real for a single fill;
In one plot we can see different curves for iRing26 fills (three different fills were taken in consideration)
these are the predictions made using different iRings for the same fill.
 
## Work in progress...
The next step is to plot the transparency over an entire run using for each fill the parameters obtained for the same fill by fitting another iRing.
First we need to know the parameter array for every fill_num in the run, then we will use it for making prediction.

