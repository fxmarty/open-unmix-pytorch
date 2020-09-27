import argparse
import musdb
import museval
import test
import multiprocessing
import functools
from pathlib import Path
import torch

import os
import json
import tqdm
from SDR_metrics import sisdr_framewise
from SDR_metrics import ideal_SDR_framewise
import numpy as np

from utils import memory_check

def evalTargets(joint,args,device):
    mus = musdb.DB(
        root=args.root,
        download=args.root is None,
        subsets=args.subset,
        is_wav=args.is_wav
    )
    
    SI_SDRscores_vocals = []
    SDRscores_vocals = []
    SI_SDRscores_accompaniment = []
    SDRscores_accompaniment = []
    tracks = []
    
    if not os.path.exists(args.evaldir):
        os.makedirs(args.evaldir)
    
    for track in tqdm.tqdm(mus.tracks):
    #if True:
        #track = mus.tracks[35]
        print(track.name)
        print(track.duration)
        print("audio shape",track.audio.shape)
        phoneme = np.load(args.root_phoneme+'/'
                                +'test'+'_'+track.name+'.npy')
        phoneme = torch.from_numpy(phoneme)
        
        if args.fake:
            phoneme = torch.zeros(phoneme.shape)
            phoneme[...,0] = 1
        
        
        estimates = test.separate(
            audio=track.audio, # shape [nb_time_points, 2]
            phoneme=phoneme,
            targets=args.targets,
            model_name=args.model,
            niter=args.niter,
            alpha=args.alpha,
            softmask=args.softmask,
            device=device
        )
        
        if args.outdir:
            mus.save_estimates(estimates, track, args.outdir)

        vocals = torch.from_numpy(track.targets['vocals'].audio.T)
        estimated_vocals = torch.from_numpy(estimates['vocals'].T)
        
        if vocals.shape != estimated_vocals.shape:
            raise ValueError("Targets and estimates should have the same shape!")

        vocals_SISDR = sisdr_framewise(estimated_vocals,vocals,
                                        sample_rate=track.rate,eps=0)
        vocals_SDR = ideal_SDR_framewise(estimated_vocals,vocals,sample_rate=track.rate)
        
        # mean over channels
        vocals_SISDR = torch.mean(vocals_SISDR,dim=1)[0]
        vocals_SDR = torch.mean(vocals_SDR,dim=1)[0]
        
        if joint:
            accompaniment = torch.from_numpy(track.targets['accompaniment'].audio.T)
            estimated_accompaniment = torch.from_numpy(estimates['accompaniment'].T)
            accompaniment_SISDR  = sisdr_framewise(estimated_accompaniment,accompaniment,
                                            sample_rate=track.rate,eps=0)
            accompaniment_SDR  = ideal_SDR_framewise(estimated_accompaniment,accompaniment,
                                            sample_rate=track.rate)
            
            accompaniment_SISDR  = torch.mean(accompaniment_SISDR,dim=1)[0]
            accompaniment_SDR  = torch.mean(accompaniment_SDR,dim=1)[0]
        
        frame_list = []
        
        for i,k in enumerate(vocals_SISDR):
            if joint:
                frame_list.append({"time" : float(i), "duration" : 1.0,
                            "metrics" : {"SI-SDR_vocals" : -vocals_SISDR[i].item(),
                            "SI-SDR_accompaniment" : -accompaniment_SISDR[i].item(),
                            "SDR_vocals" : -vocals_SDR[i].item(),
                            "SDR_accompaniment" : -accompaniment_SDR[i].item()}})
            else:
                frame_list.append({"time" : float(i), "duration" : 1.0,
                            "metrics" : {"SISDR_vocals" : -vocals_SISDR[i].item(),
                            "SDR_vocals" : -vocals_SDR[i].item()}})
        
        # median over windows
        vocals_SISDR = vocals_SISDR[torch.isfinite(vocals_SISDR)]
        vocals_SDR = vocals_SDR[torch.isfinite(vocals_SDR)]
        
        median_SISDR_vocals = np.median(vocals_SISDR.numpy())
        median_SDR_vocals = np.median(vocals_SDR.numpy())
        
        SI_SDRscores_vocals.append(-median_SISDR_vocals)
        SDRscores_vocals.append(-median_SDR_vocals)
        
        medians = [{"median_SISDR_vocals" : -median_SISDR_vocals,
                    "median_SDR_vocals" : -median_SDR_vocals}]
        
        if joint:
            accompaniment_SISDR = accompaniment_SISDR[torch.isfinite(accompaniment_SISDR)]
            accompaniment_SDR = accompaniment_SDR[torch.isfinite(accompaniment_SDR)]
            
            median_SISDR_accompaniment = np.median(accompaniment_SISDR.numpy())
            median_SDR_accompaniment = np.median(accompaniment_SDR.numpy())
            
            SI_SDRscores_accompaniment.append(-median_SISDR_accompaniment)
            SDRscores_accompaniment.append(-median_SDR_accompaniment)
            
            medians[0]["median_SISDR_accompaniment"] = -median_SISDR_accompaniment
            medians[0]["median_SDR_accompaniment"] = -median_SDR_accompaniment
                
        with open(args.evaldir+'/'+track.name+'.json', 'w') as outfile:
            json.dump([medians,frame_list], outfile, indent=2)
        
        tracks.append(track.name)
        
        torch.cuda.empty_cache()
    
    return SI_SDRscores_vocals,SDRscores_vocals,SI_SDRscores_accompaniment,SDRscores_accompaniment,tracks


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(
        description='MUSDB18 Evaluation',
        add_help=False
    )

    parser.add_argument(
        '--targets',
        nargs='+',
        default=['vocals', 'drums', 'bass', 'other'],
        type=str,
        help='provide targets to be processed. \
              If none, all available targets will be computed'
    )

    parser.add_argument(
        '--model',
        default='umxhq',
        type=str,
        help='path to mode base directory of pretrained models'
    )

    parser.add_argument(
        '--outdir',
        type=str,
        help='Results path where audio evaluation results are stored'
    )

    parser.add_argument(
        '--evaldir',
        type=str,
        help='Results path for museval estimates'
    )

    parser.add_argument(
        '--root',
        type=str,
        help='Path to MUSDB18'
    )
    
    parser.add_argument('--root-phoneme',
        type=str,
        help='root path of .pt phonemes, at acoustic model resolution'
    )

    parser.add_argument(
        '--subset',
        type=str,
        default='test',
        help='MUSDB subset (`train`/`test`)'
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA inference'
    )

    parser.add_argument(
        '--is-wav',
        action='store_true', default=False,
        help='flags wav version of the dataset'
    )
    
    parser.add_argument('--fake',
                        action='store_true',
                        help='Input fake constant phoneme')

    args, _ = parser.parse_known_args()
    args = test.inference_args(parser, args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model_path = Path(args.model).expanduser()
        
    if not model_path.exists():
        raise NameError('Model path is wrong')
            # assume model is a path to a local model_name directory
    else:
        # load model from disk, there should be only one target
        with open(Path(model_path, args.targets[0] + '.json'), 'r') as stream:
            results = json.load(stream)
    
    try:
        joint = results['args']['joint']
    except:
        joint = True
    
    SI_SDRscores_vocals,SDRscores_vocals, SI_SDRscores_accompaniment,SDRscores_accompaniment,tracks = evalTargets(joint,args,device)
    
    overall_SI_SDR_vocals = np.median(np.array(SI_SDRscores_vocals))
    overall_SDR_vocals = np.median(np.array(SDRscores_vocals))
    print("SI-SDR median over windows, median over track (vocals):",overall_SI_SDR_vocals)
    print("SDR median over windows, median over track (vocals):",overall_SDR_vocals)

    text_file = open(args.evaldir+"/_overall.txt", "w")
    text_file.write("SI-SDR for vocals (median over windows, median over track): "
                    + str(overall_SI_SDR_vocals))
    text_file.write("\n")
    text_file.write("SDR for vocals (median over windows, median over track): "
                    + str(overall_SDR_vocals))
    
    if joint:
        overall_SI_SDR_accompaniment = np.median(np.array(SI_SDRscores_accompaniment))
        overall_SDR_accompaniment = np.median(np.array(SDRscores_accompaniment))
        
        print("---")
        print("SI-SDR median over windows, median over track (accompaniment):",
                overall_SI_SDR_accompaniment)
        print("SDR median over windows, median over track (accompaniment):",
                overall_SDR_accompaniment)
        
        text_file.write("\n---\n")
        text_file.write("SI-SDR for accompaniment (median over windows, median over track): " + str(overall_SI_SDR_accompaniment))
        text_file.write("\n")
        text_file.write("SDR for accompaniment (median over windows, median over track): " + str(overall_SDR_accompaniment))
        
    text_file.write("\n\n\n")
    text_file.write("-------")
    text_file.write("\n\n")
    
    for i,track in enumerate(tracks):
        text_file.write(track + "\n")
        text_file.write("SI-SDR (vocals): " + str(SI_SDRscores_vocals[i]) + "\n")
        text_file.write("SDR (vocals): " + str(SDRscores_vocals[i]) + "\n")
        if joint:
            text_file.write("SI-SDR (accompaniment): " + str(SI_SDRscores_accompaniment[i]) + "\n")
            text_file.write("SDR (accompaniment): " + str(SDRscores_accompaniment[i])+"\n")
        text_file.write("------\n")
    
    text_file.close()