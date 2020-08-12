import os
import pandas as pd
import json
import matplotlib.pyplot as plt

"""
This piece of code allows to get comprehensive metrics from a folder with .json files containing the separation metrics computed with museval module, and so for each track (e.g. in the test set of MUSDB18).
"""

# To be specified
rootdir = '/home/felix/Documents/Mines/Césure/_Stage Télécom/Code/evalDir/test/'

# One simple example
"""
with open("/home/felix/Documents/Mines/Césure/_Stage Télécom/Code/evalDir/test/Zeno - Signs.json") as jsonfile:
    data = json.load(jsonfile)


data_df = pd.DataFrame(data)

dataBon = pd.DataFrame.from_dict(data_df["targets"][0]['frames']) # vocals
metricsOnly = dataBon['metrics']

metricsOnlyDf = pd.DataFrame(metricsOnly.tolist())
"""

listOfSongs = [] # list of DataFrame, one for each track with the 4 metrics 
for filename in os.listdir(rootdir):
    if filename.endswith('.json'):
        with open(rootdir + filename) as jsonfile:
            #print(rootdir + filename)
            data = json.load(jsonfile)
        data_df = pd.DataFrame(data)
        
        dataBon = pd.DataFrame.from_dict(data_df["targets"][0]['frames']) # vocals
        metricsOnly = dataBon['metrics']
        
        metricsOnlyDf = pd.DataFrame(metricsOnly.tolist())

        listOfSongs.append(metricsOnlyDf)

std = pd.DataFrame(columns=['SDR', 'SIR', 'SAR', 'ISR'])
median = pd.DataFrame(columns=['SDR', 'SIR', 'SAR', 'ISR'])

for i,df in enumerate(listOfSongs):
    std.loc[i] = df.std().tolist() # over tracks
    median.loc[i] = df.median().tolist() # over tracks

print("Median over frames, median over tracks:")
print(median.median())

ax = median.plot.box()
plt.show()