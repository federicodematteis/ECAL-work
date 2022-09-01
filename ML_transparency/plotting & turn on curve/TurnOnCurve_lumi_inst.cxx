#include "trdstyle.C"
#include<cmath>

//devo posizionarmi nella stessa cartella del txt per farlo andare 
void TurnOnCurve_lumi_inst() {

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

    in_file.open("measured23.txt");
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


    in_file2.open("lumi23.txt");

    if (!in_file2) {
        std::cout << "error2" << std::endl;
        exit(1);
    }
    
    float x,y,z,g,d,o,t;

    while (in_file2 >> x >> y >> z >> g >> d >> o >> t){  // >> t
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

    in_file3.open("predicted23.txt");
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
    
    int nbin=300;
    float minE = 28;
    float maxE = 32;
    float threshold = 30;
    float delta_value = (maxE-minE)/nbin;

    float conteggi_sopra_blu = 0;
    float conteggi_sotto_blu = 0;
    float conteggi_sopra_rosso =0;
    float conteggi_sotto_rosso =0;

    float eventi_persi_real = 0;
    float eventi_persi_corrected = 0;

    //conteggi per turnon pesata

    //int conteggi_tot = nbin * transparency_size;
    int conteggi_tot =0;

    TCanvas* cc_turn_on = new TCanvas ("cc_turn_on", "", 800, 600);
    TH1F *h_correction = new TH1F("h_correction", "", nbin, minE, maxE);
    TH1F *h_real = new TH1F ("h_real", "", nbin, minE, maxE);
    TH1F *h_ideal = new TH1F ("h_ideal", "", nbin, minE, maxE);
    
    gRandom->SetSeed();

    //TurnOnCurve Not corrected with transparency predictions
    // we are measuring affected Energy E*T where T is different from 1.
    float lumi_inst_0;
    float lumi_int_0;

    float counts_improve_above=0;
    float counts_improve_under=0;

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

            conteggi_tot = conteggi_tot + Lumi_inst[i]/lumi_inst_0;
            


            float correction = transp_predicted_validation[i]; // compute correction 
            float value_smeared = value*transp_validation[i]/correction; // value w/ correction
            float value_smeared_real = value*transp_validation[i]; // value w/out correction
            
            sum_weight += Lumi_inst[i]/lumi_inst_0;



            if (value_smeared > threshold) {
                h_correction->Fill(value,Lumi_inst[i]/lumi_inst_0);

                if (value > 30){
                    conteggi_sopra_rosso = conteggi_sopra_rosso + Lumi_inst[i]/lumi_inst_0;
                }

                if (value < 30){
                    conteggi_sotto_rosso = conteggi_sotto_rosso + Lumi_inst[i]/lumi_inst_0;
                }


            }

            if (value_smeared_real < threshold) {
                //h_correction->Fill(value);//,lumi_inst_0/Lumi_inst[i]);

                eventi_persi_real = eventi_persi_real + Lumi_inst[i]/lumi_inst_0;
            }

            if (value_smeared < threshold) {
                //h_correction->Fill(value);//,lumi_inst_0/Lumi_inst[i]);

                eventi_persi_corrected = eventi_persi_corrected+Lumi_inst[i]/lumi_inst_0;
            }

            if (value_smeared_real > threshold) {
                h_real->Fill(value,Lumi_inst[i]/lumi_inst_0);

                if (value > 30){
                    conteggi_sopra_blu = conteggi_sopra_blu + Lumi_inst[i]/lumi_inst_0;
                }

                if (value < 30){
                    conteggi_sotto_blu = conteggi_sotto_blu + Lumi_inst[i]/lumi_inst_0;
                }


            }

	        if (value > threshold) {
		        h_ideal->Fill(value,Lumi_inst[i]/lumi_inst_0);
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

    h_correction->SetLineWidth(2.);
    h_correction->SetLineColor(kOrange+2);
    h_correction->SetStats(0);
    h_correction->SetFillColor(kOrange+2);
    h_correction->SetFillStyle(3003);


    h_correction->Draw("histo same");
    h_correction->GetXaxis()->SetTitle("Energy [GeV]");
    h_correction->GetYaxis()->SetTitle("Trigger Efficiency");



    h_real->SetLineWidth(2.);
    h_real->SetLineColor(kCyan+3);
    h_real->SetStats(0);
    h_real->SetFillColor(kCyan+3);
    h_real->SetFillStyle(3003);


    h_real->Draw("histo same");
    h_real->GetXaxis()->SetTitle("Energy [GeV]");
    h_real->GetYaxis()->SetTitle("Trigger Efficiency");
  
 
    TLegend *legend = new TLegend();
    legend->AddEntry(h_real,"Without correction");
    legend->AddEntry(h_correction,"With correction");
    legend->AddEntry(h_ideal, "Ideal");
    legend->SetHeader("test dataset i-Ring 28");
    legend->Draw();


    
    // Derivatives of the turn on curves
    //
    Double_t x_real[nbin], y_real[nbin], x_correction[nbin], y_correction[nbin], d_real[nbin], d_correction[nbin];

    for (int i = 1; i < nbin+1; i++) {
        x_real[i] = h_real->GetBinCenter(i);
        y_real[i] = h_real->GetBinContent(i);

        x_correction[i] = h_correction->GetBinCenter(i);
        y_correction[i] = h_correction->GetBinContent(i);
    }
    // calcolare da qui la percentuale di eventi selezionati 
    for (int i = 1; i < nbin+1; i++) {
        d_real[i] = (y_real[i]-y_real[i-1])/(x_real[i]-x_real[i-1]);
        d_correction[i] = (y_correction[i]-y_correction[i-1])/(x_correction[i]-x_correction[i-1]);
    }

    Double_t point_real[nbin-8], point_correction[nbin-8];
    double sum_real = 0.;
    double sum_correction = 0.;

    for (int i = 2; i < nbin; i++) {
        sum_real = 0.;
        sum_correction = 0.;
        sum_real = d_real[i-1]+d_real[i]+d_real[i+1]; //3 points average to smooth a little
        sum_correction = d_correction[i-1]+d_correction[i]+d_correction[i+1];
        point_real[i] = sum_real/3;
        point_correction[i] = sum_correction/3;
    }
    //voglio arrivare a metà dei bin (cioè al valore di threshold 30 GeV)
    // for (int i = 0; i < (nbin)/2 + 1; i++){
    //     //se l'improv è negativo significa che vengono contati meno eventi di quelli misurati 
    //     //se positivo ne conto di più
    //     counts_improve_under = counts_improve_under + ((y_correction[i]-y_real[i]));
        
    // }

    // //parto dal bin di threshold e conto fino al bin massimo
    // for (int i = (nbin)/2 + 1 ; i < nbin+1; i++){
    //     //se l'improv è negativo significa che vengono contati meno eventi di quelli misurati 
    //     //se positivo ne conto di più
    //     counts_improve_above = counts_improve_above + ((y_correction[i]-y_real[i]));
        
    // }

    TCanvas *c_point = new TCanvas("c_point", "", 800, 700);
    TGraph *real_point = new TGraph(nbin, x_real, point_real);
    TGraph *correction_point = new TGraph(nbin, x_correction, point_correction);
    TMultiGraph *mg2 = new TMultiGraph();

    real_point->SetMarkerSize(1.);
    real_point->SetMarkerStyle(20);
    real_point->SetMarkerColor(kCyan+3);
    real_point->SetLineColor(kCyan+3);
    real_point->SetLineWidth(2);
    mg2->Add(real_point);

    correction_point->SetMarkerSize(1.);
    correction_point->SetMarkerStyle(20);
    correction_point->SetMarkerColor(kOrange+2);
    correction_point->SetLineColor(kOrange+2);
    correction_point->SetLineWidth(2);
    mg2->Add(correction_point);

    mg2->Draw("AP");
    mg2->GetXaxis()->SetLimits(28.5,31.5);
    mg2->SetMinimum(-1e-3);
    mg2->GetXaxis()->SetTitle("Energy [GeV]");
    mg2->GetXaxis()->SetLabelSize(.045);
    mg2->GetYaxis()->SetTitle("Efficiency derivative [GeV^{-1}]");

    TLegend *p_legend = new TLegend();
    p_legend->AddEntry(real_point,"Without correction");
    p_legend->AddEntry(correction_point,"With correction");
    p_legend->SetHeader("test dataset i-Ring 28");
    p_legend->Draw();

    sum_real = 0.;
    sum_correction = 0.;

    for (int i = 1; i < nbin+1; i++) {
        sum_real += d_real[i];
        sum_correction += d_correction[i];
    }
    double mean_real = sum_real/nbin;
    double mean_correction = sum_correction/nbin;

    // Variance computation
    double var_real = 0.;
    double var_correction = 0.;
    double weight_real = 0.;
    double weight_correction = 0.;
    for (int i = 1; i < nbin+1; i++) {
        var_real += point_real[i]*(x_real[i]-30.)*(x_real[i]-30.);
        weight_real += point_real[i];

        var_correction += point_correction[i]*(x_correction[i]-30.)*(x_correction[i]-30.);
        weight_correction += point_correction[i];
    }
    var_real = var_real/weight_real;
    var_correction = var_correction/weight_correction;

    std::cout << "Without correction: " << var_real << std::endl;
    std::cout << "With correction: " << var_correction << std::endl;

    std::cout << "events above threshold real " << conteggi_sopra_blu /conteggi_tot << std::endl;
    std::cout << "events under threshold real " << conteggi_sotto_blu /conteggi_tot << std::endl;
    std::cout << "eventi persi reale " << eventi_persi_real /conteggi_tot << std::endl;

    std::cout << "events above threshold corrected: " << conteggi_sopra_rosso /conteggi_tot << std::endl;
    std::cout << "events under threshold corrected " << conteggi_sotto_rosso /conteggi_tot << std::endl;
    std::cout << "eventi persi reale " << eventi_persi_real /conteggi_tot << std::endl;
    std::cout << "eventi persi reale " << eventi_persi_corrected /conteggi_tot << std::endl;

}

