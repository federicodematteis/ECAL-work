#include "trdstyle.C"
#include<cmath>

//devo posizionarmi nella stessa cartella del txt per farlo andare 
void TurnOnCurveML() {

    gStyle->SetOptStat(0);
    setTDRStyle();
    std::ifstream in_file;
    std::ifstream in_file2;
    std::ifstream in_file3;
    //measured and predicted transparency
    std::vector<float> transp_validation;
    std::vector<float> transp_predicted_validation;

    //Luminosity metadata
    std::vector<float> Lumi_inst;
    std::vector<float> Lumi_int_LHC;
    std::vector<float> Lumi_in_fill;
    std::vector<float> Lumi_last_fill;
    std::vector<float> time_in_fill;
    std::vector<float> Lumi_last_point;
    std::vector<float> Ring_index;

    float measured;

    in_file.open("transp_validation.txt");
    if (!in_file) {
        std::cout << "error1" << std::endl;
        exit(1);
    }
  
    while (in_file >> measured) {
        transp_validation.push_back(measured);

    }
    for (int i=0; i<transp_validation.size(); i++){
        std::cout << transp_validation[i] << std::endl;
    }


    in_file2.open("Luminosity_data_validation.txt");

    if (!in_file2) {
        std::cout << "error2" << std::endl;
        exit(1);
    }
    
    float x,y,z,g,d,o,t;

    while (in_file2 >> x >> y >> z >> g >> d >> o >> t){
        Lumi_inst.push_back(x);
        Lumi_int_LHC.push_back(y);
        Lumi_in_fill.push_back(z);
        Lumi_last_fill.push_back(g);
        time_in_fill.push_back(d);
        Lumi_last_point.push_back(o);
        Ring_index.push_back(t);

    }

    for (int i=0; i<Lumi_inst.size(); i++){
        std::cout << Lumi_inst[i] << std::endl;
    }

    in_file3.open("transp_predicted_validation.txt");
        if (!in_file3) {
            std::cout << "error" << std::endl;
            exit(1);
        }
    float predicted;
    while (in_file3 >> predicted) {
        transp_predicted_validation.push_back(predicted);

    }
    // for (int i=0; i<transp_validation.size(); i++){
    //         std::cout << transp_predicted_validation[i] << std::endl;

    // }
 
    int transparency_size = transp_validation.size();

    int nbin=80;
    float minE = 29;
    float maxE = 31;
    float threshold = 30;
    float delta_value = (maxE-minE)/nbin;

    TCanvas* cc_turn_on = new TCanvas ("cc_turn_on", "", 800, 600);
    TH1F *h_correction = new TH1F("h_correction", "", nbin, minE, maxE);
    TH1F *h_real = new TH1F ("h_real", "", nbin, minE, maxE);
    TH1F *h_ideal = new TH1F ("h_ideal", "", nbin, minE, maxE);
    
    gRandom->SetSeed();

    //TurnOnCurve Not corrected with transparency predictions
    // we are measuring affected Energy E*T where T is different from 1.
    float lumi_inst_0;
    float lumi_int_0;

    //lumi_inst_0 = Lumi_inst[0];

    for (int ibin = 0; ibin < nbin+1; ibin++) {
	    float sum_weight = 0.;
        float value = minE + (ibin+0.5)*delta_value;
//         std::cout << ibin << "/" << nbin << std::endl;
        for (int i = 0; i < transparency_size ; i++) {
            if (transp_validation[i] == 1.) {
                lumi_inst_0 = Lumi_inst[i];
            }
            y=Lumi_inst[i];
            float correction = transp_predicted_validation[i]; // compute correction 
            float value_smeared = value*transp_validation[i]/correction; // value w/ correction
            float value_smeared_real = value*transp_validation[i]; // value w/out correction
            sum_weight += lumi_inst_0/Lumi_inst[i];

            if (value_smeared > threshold) {
                h_correction->Fill(value);//,lumi_inst_0/Lumi_inst[i]);
            }
            if (value_smeared_real > threshold) {
                h_real->Fill(value);//,lumi_inst_0/Lumi_inst[i]);
            }

	        if (value > threshold) {
		        h_ideal->Fill(value);//,lumi_inst_0/Lumi_inst[i]);
	        }
        }


	double scale_real = h_real->GetBinContent(ibin)/sum_weight;
	double scale_correction = h_correction->GetBinContent(ibin)/sum_weight;
	double scale_ideal = h_ideal->GetBinContent(ibin)/sum_weight;

	h_real->SetBinContent(ibin, scale_real);
	h_correction->SetBinContent(ibin, scale_correction);
	h_ideal->SetBinContent(ibin, scale_ideal);
    }

    h_ideal->SetLineWidth(1.);
    h_ideal->SetLineColor(kBlack);
    h_ideal->SetStats(0);

    h_ideal->Draw("histo");
    h_ideal->GetXaxis()->SetTitle("Energy [GeV]");
    h_ideal->GetYaxis()->SetTitle("Trigger Efficiency");

    h_real->SetLineWidth(2.);
    h_real->SetLineColor(kBlue);
    h_real->SetStats(0);

    h_real->Draw("histo same");
    h_real->GetXaxis()->SetTitle("Energy [GeV]");
    h_real->GetYaxis()->SetTitle("Trigger Efficiency");
  
    h_correction->SetLineWidth(2.);
    h_correction->SetLineColor(kRed);
    h_correction->SetStats(0);

    h_correction->Draw("histo same");
    h_correction->GetXaxis()->SetTitle("Energy [GeV]");
    h_correction->GetYaxis()->SetTitle("Trigger Efficiency");

    TLegend *legend = new TLegend();
    legend->AddEntry(h_real,"Without correction");
    legend->AddEntry(h_correction,"With correction");
    legend->AddEntry(h_ideal, "Ideal");
    legend->SetHeader("iRing23");
    legend->Draw();

}

