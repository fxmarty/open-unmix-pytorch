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
import math

import soundfile as sf

import utils

def get_sec(time_str):
    """Get Seconds from time."""
    m, s = time_str.split(':')
    return int(m) * 60 + int(s)


def evalTargets(joint,args,device,type):
    
    root = args.root
    lyrics_folder = '/tsi/doctorants/kschulze/Datasets/MUSDB_w_lyrics/lyrics_transcripts/test'
    
    
    subfolders = ['d','n','x']
    
    SI_SDRscores_vocals = []
    SDRscores_vocals = []
    SI_SDRscores_accompaniment = []
    SDRscores_accompaniment = []
    sub_tracks_names = []
    
    if not os.path.exists(args.evaldir):
        os.makedirs(args.evaldir)
    
    for filename in tqdm.tqdm(os.listdir(lyrics_folder)):
        if filename.endswith('.txt'):
            # make file name for audio and text files of current track
            song_name_short = filename.replace('.txt','')
            song_name_short = song_name_short.split('-')
            song_name_short = song_name_short[0][0:6] + "_" + song_name_short[1][1:6]
            song_name_short = song_name_short.replace(" ", "_")
            
            phoneme = np.load(args.root_phoneme+'/'
                                    +'test'+'_'+filename.replace('.txt','.npy'))
            phoneme = torch.from_numpy(phoneme)
            
            with open(lyrics_folder+'/'+filename) as f:
                for i,line in enumerate(f):
                    line = line.rstrip()
                    if line.startswith('*'):
                        continue
                    line = line.split(' ')
                    if line[2] != type:
                        continue
                    
                    start = get_sec(line[0])
                    end = get_sec(line[1])
                    
                    # this is NOT the real start frame, but the last to one.
                    # This is for conserving the behavior as in training where
                    # we pad one frame with 0, see test.py
                    startFrame = math.floor(start/0.016)
                    
                    # last overlaping frame
                    endFrame = math.ceil(end/0.016) - 1
                    sub_phoneme = phoneme[startFrame:endFrame+1]
                    
                    # given to time_transform_posteriograms.py to adjust
                    # the end time
                    offset = start - math.floor(start/0.016)*0.016
                    
                    if args.fake:
                        sub_phoneme = torch.zeros(sub_phoneme.shape)
                        if len(phoneme.shape) == 2:
                            sub_phoneme[...,0] = 1
                    
                    path = root+'/mix/'+type+'/'+song_name_short+'_'+str(i)+'.pt'
                    mixture = torch.load(path)
                    
                    print(song_name_short+'_'+str(i)+'.pt')                    
                    
                    estimates = test.separate(
                        audio=mixture.numpy().T, # shape [nb_time_points, 2]
                        phoneme=sub_phoneme,
                        targets=args.targets,
                        model_name=args.model,
                        niter=args.niter,
                        alpha=args.alpha,
                        softmask=args.softmask,
                        device=device,
                        offset=offset
                    )
                    
                    
                    if args.outdir:
                        if not os.path.exists(args.outdir):
                            os.makedirs(args.outdir)
                        if not os.path.exists(args.outdir+'/'+type):
                            os.makedirs(args.outdir+'/'+type)
                        
                        for target, estimate in list(estimates.items()):
                            target_path = (args.outdir+'/'+type
                                            +'/'+song_name_short+str(i)+'_'+target+'.wav')
                            sf.write(target_path, estimate, 44100)

                    vocals = torch.load(root+'/vocals/'+type+
                                        '/'+song_name_short+'_'+str(i)+'.pt')
                    
                    estimated_vocals = torch.from_numpy(estimates['vocals'].T)
                    
                    if vocals.shape != estimated_vocals.shape:
                        raise ValueError("Targets and estimates should"+
                                        " have the same shape!")

                    vocals_SISDR = sisdr_framewise(estimated_vocals,vocals,
                                                    sample_rate=44100,eps=0)
                    vocals_SDR = ideal_SDR_framewise(estimated_vocals,
                                                    vocals,sample_rate=44100)

                    # mean over channels
                    vocals_SISDR = torch.mean(vocals_SISDR,dim=1)[0]
                    vocals_SDR = torch.mean(vocals_SDR,dim=1)[0]

                    if joint:
                        accompaniment = torch.load(root+'/accompaniments/'+type+
                                                    '/'+song_name_short+'_'+str(i)+'.pt')
                        estimated_accompaniment = torch.from_numpy(
                                                    estimates['accompaniment'].T)
                        accompaniment_SISDR  = sisdr_framewise(
                                                    estimated_accompaniment,
                                                    accompaniment,
                                                    sample_rate=44100,
                                                    eps=0)
                        accompaniment_SDR  = ideal_SDR_framewise(
                                                    estimated_accompaniment,
                                                    accompaniment,
                                                    sample_rate=44100)
                        
                        accompaniment_SISDR  = torch.mean(accompaniment_SISDR,dim=1)[0]
                        accompaniment_SDR  = torch.mean(accompaniment_SDR,dim=1)[0]

                    frame_list = []
                    
                    for i,k in enumerate(vocals_SISDR):
                        if joint:
                            frame_list.append({
                                "time" : float(i),
                                "duration" : 1.0,
                                "metrics" : {
                                "SI-SDR_vocals" : -vocals_SISDR[i].item(),
                                "SI-SDR_accompaniment" : -accompaniment_SISDR[i].item(),
                                "SDR_vocals" : -vocals_SDR[i].item(),
                                "SDR_accompaniment" : -accompaniment_SDR[i].item()
                            }})
                        else:
                            frame_list.append({
                                "time" : float(i),
                                "duration" : 1.0,
                                "metrics" : {
                                "SISDR_vocals" : -vocals_SISDR[i].item(),
                                "SDR_vocals" : -vocals_SDR[i].item()
                            }})
                    
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
                        accompaniment_SISDR = accompaniment_SISDR[
                                                torch.isfinite(accompaniment_SISDR)]
                        accompaniment_SDR = accompaniment_SDR[
                                                torch.isfinite(accompaniment_SDR)]
                        
                        median_SISDR_accompaniment = np.median(
                                                        accompaniment_SISDR.numpy())
                        median_SDR_accompaniment = np.median(accompaniment_SDR.numpy())
                        
                        SI_SDRscores_accompaniment.append(-median_SISDR_accompaniment)
                        SDRscores_accompaniment.append(-median_SDR_accompaniment)
                        
                        medians[0]["median_SISDR_accompaniment"] = (
                                                    - median_SISDR_accompaniment)
                        medians[0]["median_SDR_accompaniment"] = (
                                                    -median_SDR_accompaniment)
                    
                    
                    directory = args.evaldir+'/'+type+'/'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    
                    with open(directory+song_name_short+str(i)+'.json', 'w') as outfile:
                        json.dump([medians,frame_list], outfile, indent=2)
                
                    sub_tracks_names.append(song_name_short+str(i))
                    
                    torch.cuda.empty_cache()
            
    return (SI_SDRscores_vocals,SDRscores_vocals,SI_SDRscores_accompaniment,
            SDRscores_accompaniment,sub_tracks_names)


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
        default='/tsi/doctorants/kschulze/Datasets/MUSDB_w_lyrics441/test/audio',
        type=str,
        help='Path to Kilian repository'
    )
    
    parser.add_argument('--root-phoneme',
        type=str,
        help='root path of .npy phonemes, at acoustic model resolution'
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
    
    """
    types
    
    x: no vocals
    n: 1 singer
    s: 2+ singers, sing the same text (but maybe different notes)
    d: 2+ singers, singing different phonemes
    """
    types = ['d','n','s']
    
    SI_SDRscores_vocals = {}
    SDRscores_vocals = {}
    SI_SDRscores_accompaniment = {}
    SDRscores_accompaniment = {}
    sub_tracks = {}
    
    
    for type in tqdm.tqdm(types):
        (SI_SDRscores_vocals[type],SDRscores_vocals[type],
        SI_SDRscores_accompaniment[type],
        SDRscores_accompaniment[type],sub_tracks[type]) = evalTargets(joint,args,device,type)
        
        
        overall_SI_SDR_vocals = np.median(np.array(SI_SDRscores_vocals[type]))
        overall_SDR_vocals = np.median(np.array(SDRscores_vocals[type]))
        
        print("("+type+") SI-SDR median over windows, median over files (vocals):",
                overall_SI_SDR_vocals)
        print("("+type+") SDR median over windows, median over files (vocals):",
                overall_SDR_vocals)
        
        
        text_file = open(args.evaldir+"/_overall.txt", "a")
        text_file.write("("+type+") SI-SDR for vocals (median over windows, median over files): "
                        + str(overall_SI_SDR_vocals))
        text_file.write("\n")
        text_file.write("("+type+") SDR for vocals (median over windows, median over files): "
                        + str(overall_SDR_vocals))
        
        if joint:
            overall_SI_SDR_accompaniment = np.median(np.array(SI_SDRscores_accompaniment[type]))
            overall_SDR_accompaniment = np.median(np.array(SDRscores_accompaniment[type]))
            
            print("---")
            print("("+type+") SI-SDR median over windows, median over files (accompaniment):",
                    overall_SI_SDR_accompaniment)
            print("("+type+") SDR median over windows, median over files (accompaniment):",
                    overall_SDR_accompaniment)
            
            text_file.write("\n-\n")
            text_file.write("("+type+") SI-SDR for accompaniment (median over windows,"
                            +" median over files): " + str(overall_SI_SDR_accompaniment))
            text_file.write("\n")
            text_file.write("("+type+") SDR for accompaniment (median over windows,"
                            " median over files): " + str(overall_SDR_accompaniment))
        
        text_file.write("\n\n")
        text_file.write("---------")
        text_file.write("\n\n")
        
    text_file.close()
    