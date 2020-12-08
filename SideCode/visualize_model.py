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
import test

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
        
    mix_path = '/tsi/doctorants/fmarty/Datasets/MUSDB18_16000wav/test/The Long Wait - Dark Horses/mixture.wav'
    
    phoneme = torch.load('/tsi/doctorants/fmarty/Posteriograms/10-12_fullPosteriogramsNoProcessing/test_The Long Wait - Dark Horses.pt')
    
    mix, sr = librosa.load(mix_path, sr=None,mono=False)
    mix = torch.from_numpy(mix)
    
    
    # we crop the end of the phoneme to fit track length,
    # because Andrea padded the end in his method
    nb_time_points = mix.shape[1]
    nb_phoneme_frames = math.ceil((nb_time_points - 512)/256 + 1)
    phoneme = phoneme[:nb_phoneme_frames]
    number_of_phonemes = phoneme.shape[1]
    
    model,hyperparameters = test.load_model(target='vocals',
                                       model_name=args.model,
                                       device=device,
                                       number_of_phonemes=number_of_phonemes)
    
    # this is because 'center=True' option from STFT adds n_fft/2 padding
    # at the beginning and end, corresponding to one phoneme frame
    phoneme = torch.cat([torch.zeros(1,number_of_phonemes),phoneme,torch.zeros(1,number_of_phonemes)],dim=0)

    # add padding to the mixture to have a complete window
    # for the last frame
    mix, padding = utils.pad_for_stft(mix,
                                        hyperparameters['args']['nhop'])
    
    # Add batch dimension
    mix = mix[None,...].to(device)
    phoneme = phoneme[None,...].to(device)
    
    i = 0
    for name, W in model.named_parameters():
        print(name,W.shape)
        """
        if name.startswith('lstm.weight') or name.startswith('phoneme_network.lstmPhoneme'):
            
            if len(W.shape) == 1:
                Wcp = torch.unsqueeze(W,0).detach().cpu().numpy()
            else:
                Wcp =W.detach().cpu().numpy()
            
            plt.imshow(Wcp,interpolation='none',aspect='auto')
            plt.colorbar()
            plt.savefig(str(i) + '_'+name+'.png',dpi=1200,bbox_inches='tight')
            plt.close("all")
            plt.clf()
            i = i + 1
        """
    
    res = model(mix,phoneme)   
    dot = make_dot(res,params={**{'inputs': mix}, **dict(model.named_parameters())})
    resize_graph(dot,size_per_element=1,min_size=20)
    dot.render("attached", format="png",cleanup=True)
    
    #print(pytorch_model_summary.summary(model, mix, show_input=False))
    
    #summary(model, input_data=(2, 262144))