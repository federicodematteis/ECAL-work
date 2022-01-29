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

metadata = pd.read_csv(f"{data_folder}/output_metadata_2017_10min.csv")

data = np.load(f"{data_folder}/iRing24.npy")
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
                
                hist0.Fill(value, lumi_inst[0]/lumi_inst[i])

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

transparency = transp_func([lumi_in_fill*(1e-9), lumi_inst*(1e-9), lumi_inst_0*(1e-9)], *par_1)

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
  
hist2.SetLineWidth(2)
hist2.SetLineColor(632)
  
hist2.Draw("histo")
hist2.GetXaxis().SetTitle("Energy [GeV]")
hist2.GetYaxis().SetTitle("Efficiency")


c1.Draw()
vertical_line = ROOT.TLine(threshold, 0.0, threshold, 1.1)
vertical_line.Draw()
c1.SaveAs("h_turn_on3.png")

#commenti: fa il load di diversi iring perchè usa lo stesso codice modificando l'input

