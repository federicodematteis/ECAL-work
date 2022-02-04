## plots
Plots named Transp_prediction_entire_run_num are plot of Transparency vs time predicted and measured on the entire "run time"
These plots were obtained using the script EE_2D_Fitting : it fits the fit_func_2 with transparency data in a selected iRing (23-24-25-26)
then uses the parameters fitting for plotting the fit_func_2 as a prediction for Transparency.
So in red we see the predicted value for Transparency and in blue the measured value, and have an idea of their behaviour over time.
  
Plots named predictionfill<num> are the plots for Transparency predicted and real for a single fill;
In one plot we can see different curves for iRing26 fills (three different fills were taken in consideration)
these are the predictions made using different iRings for the same fill.
 
## Work in progress...
The next step is to plot the transparency over an entire run using for each fill the parameters obtained for the same fill by fitting another iRing.
First we need to know the parameter array for every fill_num in the run, then we will use it for making prediction.

