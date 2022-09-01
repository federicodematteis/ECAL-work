import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data_folder = ('/home/federico/root/root-6.24.06-install')
metadata = pd.read_csv(f"{data_folder}/fill_metadata_2017_10min.csv")

#Load data
data25 = np.load(f"{data_folder}/iRing23new.npy")
data24 = np.load(f"{data_folder}/iRing24new.npy")
data26 = np.load(f"{data_folder}/iRing26new.npy")
data_df25 = pd.DataFrame(data25)
data_df24 = pd.DataFrame(data24)
data_df26 = pd.DataFrame(data26)

mean25 = []
mean24 = []
mean26 = []
for i in range(0, len(data_df25.axes[1])):
    mean25 = np.append(mean25, np.mean(data_df25[i]))
    mean24 = np.append(mean24, np.mean(data_df24[i]))
    mean26 = np.append(mean26, np.mean(data_df26[i]))

#Filter data and metadata
mean25 = mean25[mean25 != -1]
mean24 = mean24[mean24 != -1]
mean26 = mean26[mean26 != -1]
metadata = metadata.iloc[:len(mean25)][mean25 != -1]

metadata_fill = metadata[metadata.in_fill == 1]

fill = metadata_fill["fill_num"].unique()

mean_fill25 = []
mean_fill24 = []
mean_fill26 = []
for f in fill :
    df = metadata_fill[metadata_fill["fill_num"] == f]
    index = df.index.values
    transp_fill25 = [mean25[i] for i in index]
    transp_fill25 = transp_fill25/transp_fill25[0]
    transp_fill24 = [mean24[i] for i in index]
    transp_fill24 = transp_fill24/transp_fill24[0]
    transp_fill26 = [mean26[i] for i in index]
    transp_fill26 = transp_fill26/transp_fill26[0]
    mean_fill25 = np.append(mean_fill25, transp_fill25)
    mean_fill24 = np.append(mean_fill24, transp_fill24)
    mean_fill26 = np.append(mean_fill26, transp_fill26)

headers = metadata_fill.columns.tolist()

for h in headers:
    correlation_matrix = np.corrcoef(mean_fill25, metadata_fill[h])
    correlation_coefficient = correlation_matrix[0][1]
    print(h + " : " + str(correlation_coefficient))

for h in headers:
    correlation_matrix = np.corrcoef(mean_fill24, metadata_fill[h])
    correlation_coefficient = correlation_matrix[0][1]
    print(h + " : " + str(correlation_coefficient))

for h in headers:
    correlation_matrix = np.corrcoef(mean_fill26, metadata_fill[h])
    correlation_coefficient = correlation_matrix[0][1]
    print(h + " : " + str(correlation_coefficient))

