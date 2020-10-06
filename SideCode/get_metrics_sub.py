import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import numpy as np
import matplotlib.patheffects as path_effects

"""
This piece of code allows to get comprehensive metrics from a folder with .json files containing the separation metrics computed with museval module, and so for each track (e.g. in the test set of MUSDB18).

Note that museval yields a single value for each metric at each window, even if the signal is stereo.
"""
def fill_df(all_types_all_exps,rootdirs,exp_names):
    """
    types
    
    x: no vocals
    n: 1 singer
    s: 2+ singers, sing the same text (but maybe different notes)
    d: 2+ singers, singing different phonemes
    """
    types = ['d','n','s']
    
    for i,rootdir in enumerate(rootdirs):
        metrics = {}
        
        for subtype in types:
            metrics[subtype] = {} # initialization for one subtype
            first_file = True
            for filename in sorted(os.listdir(rootdir+'/'+subtype)):
                if filename.endswith('.json'):
                    with open(rootdir+'/'+subtype+'/' + filename) as jsonfile:
                        #print(rootdir + filename)
                        data = json.load(jsonfile)
                    
                    data_df = pd.DataFrame(data)
                    
                    # we initialize each metric name with an empty list
                    if first_file:
                        metrics_keys = list(data_df[0][0])
                        for key in metrics_keys:
                            metrics[subtype][key] = []
                        first_file = False
                    
                    # we use the median over the file already stored in the .json
                    for key in metrics_keys:
                        metrics[subtype][key].append(data_df[0][0][key])
        
        metrics_df = pd.DataFrame.from_dict(metrics)    
        
        all_types = pd.DataFrame(columns=['metric','subset_type','value','exp','exp_number','file_number'])
        
        for subtype in types:
            # add a column for the metrics name
            subdf = metrics_df[subtype].to_frame().explode(subtype)
            
            # add index column and rename wrongly named 'index' column to 'metric'
            subdf = subdf.reset_index() 
            subdf = subdf.rename(columns = {"index" : "metric"})
            
            # value_vars goes into a column, while id_vars stays as a column renamed
            # as 'value'
            subdf = pd.melt(subdf, id_vars=['metric'], 
                            value_vars=[subtype],var_name='subset_type',
                            value_name='value')
            
            subdf['file_number'] = subdf.index
            
            # to avoid unexplained bug "ValueError: List of boxplot statistics and 
            # `positions` values must have same the length"
            # see https://stackoverflow.com/questions/42437711/numpy-memap-pandas-dataframe-and-seaborn-boxplot-troubles
            subdf.to_csv('tempo.csv',index=False)
            subdf = pd.read_csv('tempo.csv')
            
            all_types = pd.concat([all_types,subdf])
            
        all_types['exp'] = exp_names[i]
        all_types['exp_number'] = i
        all_types_all_exps = pd.concat([all_types,all_types_all_exps])
    
    os.remove('tempo.csv')
    all_types_all_exps['exp_number'] = all_types_all_exps['exp_number'].astype(int)
    return all_types_all_exps

def create_median_labels(ax, has_fliers,has_mean):
    lines = ax.get_lines()
    # depending on fliers, toggle between 5 and 6 lines per box
    lines_per_box = 5 + int(has_fliers) + int(has_mean)
    # iterate directly over all median lines, with an interval of lines_per_box
    # this enables labeling of grouped data without relying on tick positions
    for median_line in lines[4:len(lines):lines_per_box]:
        # get center of median line
        mean_x = sum(median_line._x) / len(median_line._x)
        mean_y = sum(median_line._y) / len(median_line._y)
        # print text to center coordinates
        text = ax.text(mean_x, mean_y+0.5, f'{mean_y:.1f}',
                       ha='center', va='center',
                       fontweight='regular', size=12, color='white')
        # create small black border around white text
        # for better readability on multi-colored boxes
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal(),
        ])

def show_boxplot(df,metric_list):
    metric_list = all_types_all_exps.metric.unique()
    
    for metric_name in metric_list:
        filtered_df = all_types_all_exps[all_types_all_exps['metric'] == metric_name]
        
        # mean over each file
        filtered_df = filtered_df.groupby(['metric','exp','subset_type','file_number'],as_index = False)['value'].mean()
        plt.figure()
        
        showfliers = False
        showmeans = True
        sns.set_style("whitegrid")
        box_plot = sns.boxplot(x='subset_type', y='value',hue='exp',
                        data=filtered_df,
                        whis=[10, 90],
                        flierprops = dict(markerfacecolor = '0.50',
                                        markersize = 4,marker='x'),
                        showfliers=showfliers,
                        showmeans=showmeans,
                        meanprops={"marker":"d",
                            "markerfacecolor":"red", 
                            "markeredgecolor":"black",
                            "markersize":"5"},
                        #color="w",
                        width=0.8
                        )       
        create_median_labels(box_plot.axes, showfliers, showmeans)
        plt.title(metric_name)
        plt.savefig(metric_name+'.png', dpi=300)
    #plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
    #plt.setp(ax.lines, color='k')
    
    #plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Get metrics from json files',
        add_help=False
    )
    
    parser.add_argument(
        '--root','-r',
        type=str,
        action='append',
        help='Path to the directory with .json files (e.g. test directory)',
        #required=True
    )
    
    parser.add_argument(
        '--name', '-n',
        type=str,
        action='append',
        help='Experiment name corresponding to the previous root folder',
        #required=True
    )

    args, _ = parser.parse_known_args()
    
    rootdirs = args.root
    exp_names = args.name
    
    #rootdirs = ['/home/felix/Documents/Mines/Césure/_Stage Télécom/Code/evalDir_sub','/home/felix/Documents/Mines/Césure/_Stage Télécom/Code/evalDir_sub_bis','/home/felix/Documents/Mines/Césure/_Stage Télécom/Code/evalDir_sub']
    
    #exp_names = ["Phoneme", "Fake", "Fake"]
    
    pd.set_option('display.max_columns', None)
    duplicate = len(exp_names) != len(set(exp_names))
    
    all_types_all_exps = pd.DataFrame(columns=['metric','subset_type',
                                                'value','exp'])
    
    
    all_types_all_exps = fill_df(all_types_all_exps,rootdirs,exp_names)
    
    
    #pd.set_option('display.max_rows', 50)
    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(all_types_all_exps)
    """
    show_boxplot(all_types_all_exps,exp_names)