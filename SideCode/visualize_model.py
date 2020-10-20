import torch
import numpy as np

from torchviz import make_dot

from pathlib import Path
import argparse
import json

import librosa

import math
import matplotlib.pyplot as plt


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
sys.path.insert(0,parentdir+'/models/')

import open_unmix
import utils

def load_model(target, model_name='umxhq', device='cpu'):
    """
    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)
    """
    model_path = Path(model_name).expanduser()

    if not model_path.exists():
        raise NameError('Model path is wrong')
            # assume model is a path to a local model_name directory
    else:
        # load model from disk
        with open(Path(model_path, target + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = next(Path(model_path).glob("%s*.pth" % target))
        state = torch.load(
            target_model_path,
            map_location=device
        )
        
        if results['args']['modelname'] == 'open-unmix':
            unmix = open_unmix.OpenUnmix(
                normalization_style=results['args']['normalization_style'],
                n_fft=results['args']['nfft'],
                n_hop=results['args']['nhop'],
                nb_channels=results['args']['nb_channels'],
                hidden_size=results['args']['hidden_size']
            )
            unmix.stft.center = True
            unmix.phoneme_network.center = True
            
        unmix.load_state_dict(state) # Load saved model
        unmix.eval()
        unmix.to(device)
            
        return unmix,results
        
def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.
    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)



def plot_grad_flow(named_parameters,i):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    fig, ax = plt.subplots()
    plt.plot(ave_grads, color="blue",linewidth=0.7)
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=0.5, color="blue" )
    #plt.rcParams['xtick.labelsize']=4
    plt.xticks(range(0,len(ave_grads), 1), layers,rotation="vertical",fontsize=3)
    #plt.rcParams['xtick.labelsize']=4
    plt.xlim(xmin=0, xmax=len(ave_grads))
    """
    for xc in range(len(ave_grads)):
        plt.axvline(x=xc)
    """
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('TESTGRAD'+str(i)+'.png',dpi=600,bbox_inches='tight',pad_inches=0.7)
    plt.close("all")
    plt.clf()
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(
                        description='A tool to hopefully debug models in PyTorch')
    
    parser.add_argument('--model', type=str, help='Path to model folder')
    
    args, _ = parser.parse_known_args()
    
    model,hyperparameters = load_model(target='vocals',
                                       model_name=args.model,
                                       device=device)
    
    #print(model)
    
    mix_path = '/tsi/doctorants/fmarty/Datasets/MUSDB18_16000wav/test/The Long Wait - Dark Horses/mixture.wav'
    
    phoneme = torch.load('/tsi/doctorants/fmarty/Posteriograms/10-12_fullPosteriogramsNoProcessing/test_The Long Wait - Dark Horses.pt')
    
    mix, sr = librosa.load(mix_path, sr=None,mono=False)
    mix = torch.from_numpy(mix)
    
    
    # we crop the end of the phoneme to fit track length,
    # because Andrea padded the end in his method
    nb_time_points = mix.shape[1]
    nb_phoneme_frames = math.ceil((nb_time_points - 512)/256 + 1)
    phoneme = phoneme[:nb_phoneme_frames]
    
    # this is because 'center=True' option from STFT adds n_fft/2 padding
    # at the beginning and end, correspondig to one phoneme frame
    phoneme = torch.cat([torch.zeros(1,65),phoneme,torch.zeros(1,65)],dim=0)
    
    # add padding to the mixture to have a complete window
    # for the last frame
    mix, padding = utils.pad_for_stft(mix,
                                        hyperparameters['args']['nhop'])
    
    # Add batch dimension
    mix = mix[None,...].to(device)
    phoneme = phoneme[None,...].to(device)
    
    res = model(mix,phoneme)   
    dot = make_dot(res,params={**{'inputs': mix}, **dict(model.named_parameters())})
    resize_graph(dot,size_per_element=1,min_size=20)
    dot.render("attached", format="png",cleanup=True)
    
    #print(pytorch_model_summary.summary(model, mix, show_input=False))
    
    #summary(model, input_data=(2, 262144))