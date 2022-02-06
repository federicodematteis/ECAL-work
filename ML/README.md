## in the script transparency_regression.py: 

Regression with a 2-fully-connected-layer dnn, using one iRing as a train dataset and another one as a validation dataset;
the chosen activation function is relu for both layers, the error function (the loss) is chosen as MSE (mean square error).
Plot loss function and predicted transparency vs time, on the same plot draw real transparency data; 
This plot needs a comparison with the same plot obtained from the previous work (~/plotting folder).

## Work in progress
Now i want to use more than 1 iRing.npy as train dataset; 
the first purpose is to append one luminosity dataset to its own queue and use the new duplicated metadata-frame as inputs for training; the second is to take the time data into consideration as an input, in order to predict transparency data on a single iRing.


