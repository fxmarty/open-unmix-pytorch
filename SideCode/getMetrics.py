import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

"""
This piece of code allows to get comprehensive metrics from a folder with .json files containing the separation metrics computed with museval module, and so for each track (e.g. in the test set of MUSDB18).

Note that museval yields a single value for each metric at each window, even if the signal is stereo.
"""

parser = argparse.ArgumentParser(
    description='Get metrics from json files',
    add_help=False
)

parser.add_argument(
    '--rootdir',
    type=str,
    help='Path to the directory with .json files (e.g. test directory)'
)

args, _ = parser.parse_known_args()
rootdir = args.rootdir

# To be specified
#rootdir = '/home/felix/Documents/Mines/Césure/_Stage Télécom/Code/evalDir/test/'
number_targets = 0
target_names = []

# list of list of DataFrame, one for each track with the 4 metrics for each target
listOfSongs = [] 
for filename in os.listdir(rootdir):
    if filename.endswith('.json'):
        with open(rootdir + filename) as jsonfile:
            #print(rootdir + filename)
            data = json.load(jsonfile)
        data_df = pd.DataFrame(data)
        number_targets = len(data_df["targets"])
        metrics_song = []
        for i in range(number_targets):
            #print(len(data_df["targets"]))
            dataBon = pd.DataFrame.from_dict(data_df["targets"][i]['frames'])
            
            if len(target_names) < number_targets:
                target_names.append(data_df["targets"][i]['name'])
            
            metricsOnly = dataBon['metrics']
            
            # DataFrame with metrics for one song at each window
            metrics_target = pd.DataFrame(metricsOnly.tolist()) 
    
            metrics_song.append(metrics_target)
            #print(metrics_target)
        listOfSongs.append(metrics_song)



std = [pd.DataFrame(columns=['SDR', 'SIR', 'SAR', 'ISR']) for k in range(number_targets)]
median = [pd.DataFrame(columns=['SDR', 'SIR', 'SAR', 'ISR']) for k in range(number_targets)]

for i,metrics_song in enumerate(listOfSongs):
    for k,df in enumerate(metrics_song):
        std[k].loc[i] = df.std().tolist() # over frames in a track
        median[k].loc[i] = df.median().tolist() # over frames in a track

for i,target in enumerate(median):
    print("Median over frames, median over tracks ("+target_names[i]+"):")
    print(median[i].median()) # over tracks
    sns.set_style("whitegrid") 
    sns.boxplot(x="variable", y="value",data=pd.melt(median[i]),whis=[5, 95],flierprops = dict(markerfacecolor = '0.50', markersize = 4,marker='x')) 
    plt.title("Metrics value for MUSDB18 (target "+target_names[i]+")")
    plt.savefig('petittest'+str(i)+'.png',dpi=300)
    plt.close("all")
    plt.clf()
#plt.show()
