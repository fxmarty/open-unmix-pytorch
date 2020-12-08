import argparse
import musdb
import museval
import test
import multiprocessing
import functools
from pathlib import Path
import torch

import soundfile as sf
import os
import json
import tqdm
from SDR_metrics import sdr
from SDR_metrics import ideal_SDR
import numpy as np

from utils import memory_check
import unvoiced_detection

# enforce deterministic behavior
def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        print(track.name)
        phoneme = torch.load(args.root_phoneme+'/'
                                +'test'+'_'+track.name+'.pt')
        
        frame_list = []
        
        estimates = test.separate(
            audio=track.audio, # shape [nb_time_points, 2]
            phoneme=phoneme,
            targets=args.targets,
            model_name=args.model,
            niter=args.niter,
            alpha=args.alpha,
            softmask=args.softmask,
            device=device,
            enforce_fake=args.enforce_fake
        )
        
        if args.outdir:
            if not os.path.exists(args.outdir):
                os.makedirs(args.outdir)

        vocals = torch.from_numpy(track.targets['vocals'].audio.T)
        estimated_vocals = torch.from_numpy(estimates['vocals'].T)
        
        if vocals.shape != estimated_vocals.shape:
            raise ValueError("Targets and estimates should have the same shape!")
        
        if len(vocals.shape) == 1:
            vocals_mono = vocals
        
        elif vocals.shape[1] == 2:
            vocals_mono = vocals.sum(axis=1) / 2
        
        elif vocals.shape[0] == 2:
            vocals_mono = vocals.sum(axis=0) / 2
        
        unv_start,unv_end = unvoiced_detection.detect(vocals_mono.numpy(), track.rate)
        
        if joint:
            accompaniment = torch.from_numpy(track.targets['accompaniment'].audio.T)
            estimated_accompaniment = torch.from_numpy(estimates['accompaniment'].T)
        
        saved_voc_SISDR = []
        saved_voc_SDR = []
        saved_acc_SISDR = []
        saved_acc_SDR = []
        
        for i in range(len(unv_start)):
            if args.outdir:
                if not os.path.exists(args.outdir + '/true/' + track.name):
                    os.makedirs(args.outdir + '/true/' + track.name)
                
                if not os.path.exists(args.outdir + '/estimate/' + track.name):
                    os.makedirs(args.outdir + '/estimate/' + track.name)
                
                unvoiced_end = unv_end[i]/track.rate
                unvoiced_start = unv_start[i]/track.rate
                name = f"{unvoiced_start:.3f}" + '-' + f"{unvoiced_end:.3f}" + '.wav'
                sf.write(args.outdir + '/true/' + track.name + '/'+ name,
                        track.targets['vocals'].audio[unv_start[i]:unv_end[i]], 
                        samplerate=track.rate)
                
                sf.write(args.outdir + '/estimate/' + track.name + '/'+ name,
                        estimates['vocals'][unv_start[i]:unv_end[i]], 
                        samplerate=track.rate)
                
            
            
            extract_estimate_vocals = estimated_vocals[...,unv_start[i]:unv_end[i]]
            extract_target_vocals = vocals[...,unv_start[i]:unv_end[i]]
            vocals_SISDR = sdr(extract_estimate_vocals,extract_target_vocals,
                                            eps=0,
                                            scale_invariant=True)
            vocals_SDR = ideal_SDR(extract_estimate_vocals,
                                    extract_target_vocals)
            
            if joint:
                extract_estimate_acc = estimated_accompaniment[...,unv_start[i]:unv_end[i]]
                extract_target_acc = accompaniment[...,unv_start[i]:unv_end[i]]
                
                accompaniment_SISDR  = sdr(extract_estimate_acc,
                                            extract_target_acc,
                                            eps=0)
                accompaniment_SDR  = ideal_SDR(extract_estimate_acc,
                                                extract_target_acc
                                                )
                
                duration = (unv_end[i] - unv_start[i])/track.rate
                
                frame_list.append({"time" : unv_start[i], "duration" : duration,
                            "metrics" : {"SI-SDR_vocals" : vocals_SISDR,
                            "SI-SDR_accompaniment" : accompaniment_SISDR,
                            "SDR_vocals" : vocals_SDR,
                            "SDR_accompaniment" : accompaniment_SDR}})
            
            saved_voc_SISDR.append(vocals_SISDR)
            saved_voc_SDR.append(vocals_SDR)
            saved_acc_SISDR.append(accompaniment_SISDR)
            saved_acc_SDR.append(accompaniment_SDR)
        
        
        saved_voc_SISDR = torch.tensor(saved_voc_SISDR)
        saved_voc_SDR = torch.tensor(saved_voc_SDR)
        saved_acc_SISDR = torch.tensor(saved_acc_SISDR)
        saved_acc_SDR = torch.tensor(saved_acc_SDR)
        
        # median over windows
        vocals_SISDR = saved_voc_SISDR[torch.isfinite(saved_voc_SISDR)]
        vocals_SDR = saved_voc_SDR[torch.isfinite(saved_voc_SDR)]
        
        median_SISDR_vocals = np.median(vocals_SISDR.numpy())
        median_SDR_vocals = np.median(vocals_SDR.numpy())
        
        SI_SDRscores_vocals.append(median_SISDR_vocals)
        SDRscores_vocals.append(median_SDR_vocals)
        
        medians = [{"median_SISDR_vocals (unvoiced)" : np.float64(median_SISDR_vocals),
                    "median_SDR_vocals (unvoiced)" : np.float64(median_SDR_vocals)}]
        
        accompaniment_SISDR = saved_acc_SISDR[torch.isfinite(saved_acc_SISDR)]
        accompaniment_SDR = saved_acc_SDR[torch.isfinite(saved_acc_SDR)]
        
        median_SISDR_accompaniment = np.median(accompaniment_SISDR.numpy())
        median_SDR_accompaniment = np.median(accompaniment_SDR.numpy())
        
        SI_SDRscores_accompaniment.append(median_SISDR_accompaniment)
        SDRscores_accompaniment.append(median_SDR_accompaniment)
        
        medians[0]["median_SISDR_accompaniment"] = np.float64(median_SISDR_accompaniment)
        medians[0]["median_SDR_accompaniment"] = np.float64(median_SDR_accompaniment)
        
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
    
    parser.add_argument('--enforce-fake',
                        action='store_true',
                        help='Input fake phoneme no matter the model')
    
        
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
    
    seed_all(results['args']['seed'])
    
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