import torch
import numpy as np
import argparse
import soundfile as sf
import norbert
import json
from pathlib import Path
import scipy.signal
import resampy
import math

import model

import matplotlib.pyplot as plt

import warnings
import tqdm
from contextlib import redirect_stderr
import io

def load_model(number_of_phonemes, model_name='umxhq', device='cpu'):
    """
    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)
    """
    target='model'
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
        
        autoencoder = model.Autoencoder(number_of_phonemes=number_of_phonemes,
                                    bottleneck_size=results['args']['bottleneck_size'])
        
        autoencoder.load_state_dict(state) # Load saved model
        autoencoder.eval()
        autoencoder.to(device)
            
        return autoencoder,results

def separate(
    phoneme,
    device='cpu'
):

    with torch.no_grad():
        # convert numpy audio to torch
        mixture = torch.tensor(audio.T[None, ...]).float().to(device)
        # mixture shape [1,2,nb_time_points]
        

    return estimates

def test_main(
    phoneme_path=None,
    model='umxhq',
    outdir=None, no_cuda=False,
    enforce_fake=False):

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
        
    phoneme = torch.load(phoneme_path)
    number_of_phonemes = phoneme.shape[1]
    
    # add batch dimension
    phoneme = phoneme[None,...].float().to(device)
    
    autoencoder,args = load_model(
        model_name=model,
        device=device,
        number_of_phonemes=number_of_phonemes
    )
    
    with torch.no_grad():
        bottleneck = autoencoder.encoder(phoneme)
        estimate = autoencoder.decoder(bottleneck)
        
        
        plt.imshow(phoneme[0,:,:5000].detach().cpu().numpy(),
                    aspect=0.01,interpolation='none')
        plt.colorbar()
        plt.savefig(outdir+ '/' + 'phoneme_input'+'.png',dpi=1200,bbox_inches='tight')
        plt.close("all")
        plt.clf()
        
        plt.imshow(bottleneck[0,:,:5000].detach().cpu().numpy(),
                    aspect=0.005,interpolation='none')
        plt.colorbar()
        plt.savefig(outdir+ '/' + 'bottleneck'+'.png',dpi=1200,bbox_inches='tight')
        plt.close("all")
        plt.clf()
        
        plt.imshow(estimate[0,:,:5000].detach().cpu().numpy(),
                    aspect=0.01,interpolation='none')
        plt.colorbar()
        plt.savefig(outdir+ '/' + 'reconstruction'+'.png',dpi=1200,bbox_inches='tight')
        plt.close("all")
        plt.clf()
    


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(
        description='OSU Inference',
        add_help=False
    )
    
    parser.add_argument(
        '--phoneme',
        type=str,
        help='Path to corresponding posteriogram.',
        required=True
    )

    parser.add_argument(
        '--outdir',
        type=str,
        help='Results path where audio evaluation results are stored',
        required=True
    )

    parser.add_argument(
        '--model',
        default='umxhq',
        type=str,
        help='path to mode base directory of pretrained models'
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA inference'
    )
        
    args, _ = parser.parse_known_args()
    
    test_main(
        phoneme_path=args.phoneme,
        model=args.model,
        outdir=args.outdir,
        no_cuda=args.no_cuda
        )