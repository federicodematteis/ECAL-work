import ROOT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import special

#qui vieve fatto il fit con 2 metadati: lumi_in_fill e lumi_inst
#con due forme funzionali diverse e vengono effettuati per ognuna un plot
def transp_func(data,a,b,c,d,e,f):
    x = data[0]
    y = data[1]
    y_0 = data[2]
    return (a*np.exp(-b*x)+(1-a)*np.exp(c*x))*(d*np.exp(-e*(y-y_0))+(1-d)*np.exp(f*(y-y_0)))


#la seconda funzione non viene usata in TurnOnCurve.cxx, si usa solo la prima
def transp_func_2(data,a,b,c,d,e,f):
    x = data[0]
    y = data[1]
    y_0 = data[2]
    return (a*np.exp(-b*x)+(1-a)*np.exp(c*x))*(d*np.exp(-e*y)+(1-d)*np.exp(f*y))/(d*np.exp(-e*y_0)+(1-d)*np.exp(f*y_0))

#------Load Data-----

data_folder = ('/home/federico/root/root-6.24.06-install')

metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")

data = np.load(f"{data_folder}/iRing24new.npy")
data_df = pd.DataFrame(data)

mean = []
for i in range(0, len(data_df.axes[1])):
    mean = np.append(mean, np.mean(data_df[i]))

#filter data and metadata

mean = mean[mean != -1]
metadata = metadata.iloc[:len(mean)][mean != -1]

fill = metadata["fill_num"].unique()
fill= fill[fill != 0]


metadata_fill = metadata[metadata.fill_num.isin(fill)]
metadata_fill = metadata_fill[(metadata_fill.lumi_inst >= 0.0001*1e9) & (metadata_fill.lumi_inst <= 0.0004*1e9) & (metadata_fill.lumi_in_fill >= 0.1*1e9)]

fill_num = metadata_fill.fill_num.unique()
transp_fill = []
lumi_inst_0 = []
lumi_int_0 = []

for k in fill_num:
    df = metadata_fill[metadata_fill.fill_num == k]
    transp = [mean[i] for i in df.index.values]
    transp = transp/transp[0]
    transp_fill = np.append(transp_fill, transp)
    a = np.empty(np.size(transp))
    b = np.empty(np.size(transp))
    a.fill(df['lumi_inst'].iloc[0])
    b.fill(df['lumi_int'].iloc[0])
    lumi_inst_0 = np.append(lumi_inst_0, a)
    lumi_int_0 = np.append(lumi_int_0, b)

print("hello boy")
par_1, pcov_1 = curve_fit(transp_func, [metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), lumi_inst_0*(1e-9)], transp_fill, maxfev=5000)
par_2, pcov_2 = curve_fit(transp_func_2, [metadata_fill.lumi_in_fill*(1e-9), metadata_fill.lumi_inst*(1e-9), lumi_inst_0*(1e-9)], transp_fill, maxfev=5000)
print(par_2)


#fit parameters found by fitting iRings - fill 6371
par25_6371 = [0.09192309,2.85013885,0.15955624,9.20390233,-3.23549497,0.99469655]
par26_6371 = [0.0912546,3.8466344,0.15864503,17.50818141,-3.03488584,0.84651477]
par24_6371 = [0.08717691,3.71098111,0.14912243,15.01841404,-3.06543558,0.73526334]
par23_6371 = [0.08334285,3.73442396,0.14184352,13.85424341,-3.34668462,0.91228228]
#fit parameters found by fitting iRings - fill 5962
par26_5962 = [0.0439635,9.52461282,0.06050369,23.55811403,-4.03191959,0.97508831]
par25_5962 = [9.99999958e-01,2.38597330e-02,3.63776541e+01,-2.36672498e-01,1.25178511e+01,1.00519322e+00]
par24_5962 = [0.03909184,9.11703406,0.05344768,22.0592916,-3.77427049,0.98365022]
par23_5962 = [0.037259,8.72289079,0.05183301,14.48198252,-4.85845398,0.97573511]
#fit parameters for fill 6297
par23_6297=[0.10722318,3.40142748,0.16528638,3.44148397,-14.4458373,1.00400629]
par24_6297=[0.10979645,3.46056918,0.17069329,-9.31489117,4.18862643,1.04023372]
par25_6297=[0.10798524,2.71326079,0.167472,8.46255498,-4.18165624,0.99664376]
par26_6297=[0.10798524,2.71326079,0.167472,8.46255498,-4.18165624,0.99664376]


nbin = 400
minimo = 0
massimo = 60
threshold = 30
delta_value = (massimo-minimo)/nbin

nEvents = 1000

#-----------Single Fill---------
#scegliere il fill 
print("metti input num_fill")
fill = input()

selected_metadata = metadata[metadata.fill_num == int(fill)]
selected_transp = [mean[i] for i in selected_metadata.index.tolist()]

print("trasparenza selezionata in questo fill")
print(selected_transp)
print(len(selected_transp))

lumi_in_fill = selected_metadata.lumi_in_fill.to_numpy()
lumi_inst = selected_metadata.lumi_inst.to_numpy()
lumi_inst_0 = np.empty(np.size(selected_transp))
lumi_inst_0.fill(lumi_inst[0])


#--------------------------------HREAL
c = ROOT.TCanvas("cc_turn_on", "", 800, 700)
hist0 = ROOT.TH1F("real", "", nbin, minimo, massimo)

for ibin in range(0, nbin):
    value = minimo+(ibin+0.2)*delta_value
    for iEvent in range(0, nEvents):
        for i in range(0, np.size(selected_transp)):
            

            value_smeared = value*selected_transp[i]
            
            if value_smeared > threshold:
                
                hist0.Fill(value)


hist0.Scale(1./(nEvents*np.size(selected_transp)))
  
hist0.SetLineWidth(2)
hist0.SetLineColor(632)
  
hist0.Draw("histo")
hist0.GetXaxis().SetTitle("Energy [GeV]")
hist0.GetYaxis().SetTitle("Efficiency")


c.Draw()
vertical_line = ROOT.TLine(threshold, 0.0, threshold, 1.1)
vertical_line.Draw()
c.SaveAs("h_turn_on.png")

#-------------H F1------------------

c0 = ROOT.TCanvas("cc_turn_on", "", 800, 700)
hist = ROOT.TH1F("f_1", "", nbin, minimo, massimo)

transparency = transp_func([lumi_in_fill*(1e-9), lumi_inst*(1e-9), lumi_inst_0*(1e-9)], *par23_6371)

for ibin in range(0, nbin):
    value = minimo+(ibin+0.2)*delta_value
    for iEvent in range(0, nEvents):
        for i in range(0, np.size(selected_transp)):
            value_smeared = value*transparency[i]
            if value_smeared > threshold:
                hist.Fill(value)

hist.Scale(1./(nEvents*np.size(selected_transp)))
  
hist.SetLineWidth(2)
hist.SetLineColor(632)
  
hist.Draw("histo")
hist.GetXaxis().SetTitle("Energy [GeV]")
hist.GetYaxis().SetTitle("Efficiency")



c0.Draw()
vertical_line = ROOT.TLine(threshold, 0.0, threshold, 1.1)
vertical_line.Draw()
c0.SaveAs("h_turn_on2.png")

#------------------------------------------


c1 = ROOT.TCanvas("cc_turn_on_2", "", 800, 700)
hist2 = ROOT.TH1F("f_2", "", nbin, minimo, massimo)

transparency = transp_func_2([lumi_in_fill[i]*(1e-9), lumi_inst[i]*(1e-9), lumi_inst_0[i]*(1e-9)], *par_2)


for ibin in range(0, nbin):
    value = minimo+(ibin+0.2)*delta_value
    for iEvent in range(0, nEvents):
        for i in range(0, np.size(selected_transp)):
            value_smeared = value*transparency
            if value_smeared > threshold:
                hist2.Fill(value)


hist2.Scale(1./(nEvents*np.size(selected_transp)))
  
# hist2.SetLineWidth(2)
# hist2.SetLineColor(632)
  
# hist2.Draw("histo")
# hist2.GetXaxis().SetTitle("Energy [GeV]")
# hist2.GetYaxis().SetTitle("Efficiency")


#c1.Draw()
#vertical_line = ROOT.TLine(threshold, 0.0, threshold, 1.1)
#vertical_line.Draw()
#c1.SaveAs("h_turn_on3.png")

#commenti: fa il load di diversi iring perchè usa lo stesso codice modificando l'input

