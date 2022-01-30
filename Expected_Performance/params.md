# Parameters obtained from fitting fit_func_2 with iRings transparency datas

in EE_2D_Fitting.py:
...
def fit_func2(data, a, b, c, d, e, f):
    x = data[0]
    y = data[1]
    y_0 = data[2]
    return (a*np.exp(-b*x)+(1-a)*np.exp(c*x))*(d*np.exp(-e*(y-y_0))+(1-d)*np.exp(f*(y-y_0)))
...

This file contains fit parameters for function fit_func_2 in the EE_2D_Fitting.py (~/Plotting folder) for different iRings, (to find them compile EE_2D_Fitting using .npy files in the ~Plotting folder).
We will make a copy of params and use them in TurnOnCurve.cxx (see TurnOnCurve.cxx, rows 58-64)


fit parameters                                                             |         iRing

[ 0.99926682  0.00796825  4.06367104  4.45252793  1.32586626 -2.34005698]  |		  1

[ 0.99651334  0.0346215   3.768175    3.50900096 -0.30093402 -1.37717335]  |          22

[ 0.99611681  0.03823509  3.74505104  5.17053814  0.69641096 -2.04656093]  |          23

[ 0.99577195  0.04078455  3.71418504  4.62657289  1.37968447 -3.02926379]  |          24

[ 0.9951447   0.04381734  3.60349883  7.09087354  2.71779965 -3.75657106]  |          25

[ 0.9948411   0.04640516  3.59099024 10.43638894  2.71419567 -3.37443575]  |          26

[ 0.99368691  0.05001983  3.39843497  6.00933008  2.57854487 -3.71203269]  |          27

[ 0.9930722   0.05295141  3.33611748 12.46721937  7.5420149  -8.46141772]  |          28


