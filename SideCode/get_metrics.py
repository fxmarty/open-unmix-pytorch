import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import numpy as np
import matplotlib.patheffects as path_effects
import math

"""
This piece of code allows to get comprehensive metrics from a folder with .json files containing the separation metrics, and so for each track (e.g. in the test set of MUSDB18).
"""
def fill_df(all_types_all_exps,rootdirs,exp_names,framewise):
    """
    Input:
        all_types_all_exps : pandas DataFrame
            here, an empty dataframe with column 'metric', 'value', 'exp'
        rootdirs : list
            a list of directories where the .json files with the evaluation metrics
            are stored. The key point is that several models can be evaluated at
            the same time, each with one evaluation folder with the .json.
        exp_names : list
            a list of experiment names corresponding to the paths in rootdirs.
            The box plots will be done by concatenating the metrics from different
            rootdirs with the same experiment name.
        framewise : boolean
            If True, the box plot will be done with the metrics at the frame level.
            If False, the box plot will be done with the metrics at the song level,
            that is with the median of each song already computed beforehand.
    Output:
        all_types_all_exps : pandas DataFrame
            the DataFrame from the input filled with all the experiments, with
            columns:
                'metric': the name of the metric (e.g. "SDR")
                'value': the value for the corresponding song
                'exp': the name of the experiment as specified in exp_names
                'exp_number': the number of the experiment (0, 1, 2, etc.)
                'file_number': a number corresponding to a song (0, 1, 2, etc.)
    """
    for i,rootdir in enumerate(rootdirs):
        metrics = {}
        metrics["explode_key"] = {}
        
        first_file = True
        for filename in sorted(os.listdir(rootdir)):
            if filename.endswith('.json'):
                with open(rootdir+'/'+filename) as jsonfile:
                    #print(rootdir + filename)
                    data = json.load(jsonfile)
                
                data_df = pd.DataFrame(data)
                
                if framewise == False:
                    # we initialize each metric name with an empty list
                    if first_file:
                        metrics_keys = list(data_df[0][0])
                        for key in metrics_keys:
                            metrics["explode_key"][key] = []
                        first_file = False
                    
                    # we use the median over the file already stored in the .json
                    for key in metrics_keys:
                        metrics["explode_key"][key].append(data_df[0][0][key])
                    
                elif framewise == True:
                    # we initialize each metric name with an empty list
                    if first_file:
                        metrics_keys = list(data_df.T[1][0]["metrics"].keys())
                        for key in metrics_keys:
                            metrics["explode_key"][key] = []
                        first_file = False
                    
                    # we append the metrics for each frame, median is computed later
                    for key in metrics_keys:
                        for frame in data_df.T[1]: # over frames
                            if math.isfinite(frame["metrics"][key]):
                                metrics["explode_key"][key].append(frame["metrics"][key])
        
        metrics_df = pd.DataFrame.from_dict(metrics)    
        
        all_types = pd.DataFrame(columns=['metric','value','exp','exp_number','file_number'])
        
        # add a column for the metrics name
        subdf = metrics_df["explode_key"].to_frame().explode("explode_key")
        
        print(subdf)
        
        # add index column and rename wrongly named 'index' column to 'metric'
        subdf = subdf.reset_index() 
        subdf = subdf.rename(columns = {"index" : "metric"})
        
        # value_vars goes into a column, while id_vars stays as a column renamed
        # as 'value'
        subdf = pd.melt(subdf, id_vars=['metric'], 
                        value_vars=["explode_key"],var_name='bidon',
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
    """
    A function based on https://stackoverflow.com/a/63295846/4370080 to
    add median values to a box plot in seaborn
    """
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

def show_boxplot(df,framewise,file_head):
    """
    Input:
        df : pandas DataFrame
            a dataframe with columns 'metric', 'value', 'exp', 'exp_number', 
            'file_number'
        framewise : boolean
            no computation being done here, it is used only to name the output
            .png files
        file_head : str
            a text to add to the name of the output .png files to give
    Output:
        boxplots saved as .png files, one for each metric in df dataframe, grouped
        by experiment name
    """
    metric_list = all_types_all_exps.metric.unique()
    
    for metric_name in metric_list:
        filtered_df = all_types_all_exps[all_types_all_exps['metric'] == metric_name]
        
        # mean over each file
        filtered_df = filtered_df.groupby(['metric','exp','file_number'],as_index = False)['value'].mean()

        plt.figure(figsize=[4,5])
        sns.set_style("whitegrid")
        showfliers = False
        showmeans = True
        box_plot = sns.boxplot(x='exp', y='value',#hue='exp',
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
                        width=0.8,
                        )
        
        create_median_labels(box_plot.axes, showfliers, showmeans)
        plt.title(metric_name,fontsize=12)
        plt.tight_layout()
        if framewise == False:
            plt.savefig(file_head+'_songwise_'+metric_name+'.png', dpi=300)
        elif framewise == True:
            plt.savefig(file_head+'_framewise_'+metric_name+'.png', dpi=300)
        print(".png saved.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Get metrics from json files',
        add_help=False
    )
    
    parser.add_argument(
        '--root','-r',
        type=str,
        action='append',
        help='Path to the directory with .json files, e.g. expName/evalDir_sub'
    )
    
    parser.add_argument(
        '--name', '-n',
        type=str,
        action='append',
        help='Experiment name corresponding to the previous root folder'
    )
    
    parser.add_argument(
        '--file-head',
        type=str,
        help='Head for name of files for evaluation'
    )
    
    parser.add_argument(
        '--framewise',
        action='store_true', default=False,
        help='If set, metrics will be ploted using framewise metrics to compute the boxplots and NOT the songwise metrics (already median over song)'
    )
    
    args, _ = parser.parse_known_args()
    
    rootdirs = args.root
    exp_names = args.name
    
    for i,experiment_path in enumerate(rootdirs):
        # set this folder to your own where your models are stored
        rootdirs[i] = '/tsi/doctorants/fmarty/executedJobs/'+experiment_path

    pd.set_option('display.max_columns', None)
    duplicate = len(exp_names) != len(set(exp_names))
    
    all_types_all_exps = pd.DataFrame(columns=['metric',
                                                'value','exp'])
    
    #pd.set_option('display.max_columns', None)  # or 1000
    #pd.set_option('display.max_rows', None)  # or 1000
    #pd.set_option('display.max_colwidth', -1)  # or 199

    all_types_all_exps = fill_df(all_types_all_exps,rootdirs,
                        exp_names,args.framewise)
    
    show_boxplot(all_types_all_exps,args.framewise,args.file_head)