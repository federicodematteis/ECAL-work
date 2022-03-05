hist = ROOT.TH1F("ideal-step function", "", nbin, minimo, massimo)
#transparency = transp_predicted_validation

for ibin in range(0, nbin):
    value = minimo+(ibin+0.2)*delta_value
    for iEvent in range(0, nEvents):
        for i in range(0, np.size(transparency)):
            value_smeared = value#*transparency[i]
            if value_smeared > threshold:
                hist.Fill(value)
                #hist.Eval(  , "A")


#hist.Scale(1./(nEvents*np.size(transp_predicted_validation)))
hist.SetLineWidth(2)
hist.SetLineColor(632)

#hist.SetFillColor(632)
hist.Draw("h same")
hist.SetTitle("trigger efficiency")
hist.GetXaxis().SetTitle("Energy [GeV]")
hist.GetYaxis().SetTitle("Efficiency")

