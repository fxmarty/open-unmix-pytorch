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


# enforce deterministic behavior
def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evalTargets(args,device,typee):
    """
    Input:
        joint : boolean
            joint model between target / rest or not. Argument currently not meaningful
            because Open-Unmix is NOT a joint model. However since Open-Unmix also
            estimatest the residual by substracting, default value is True
        args : argparse arguments
        device : torch.device
            set CPU or GPU depending on the machine which is used
        typee : str
            value in ("d","n","s"), that are labels for three types of extracts:
            2+ singers who sing different text, 1 singer, 2+ singers who sing the
            same text. Correspond to 3 folders /d/, /n/, /s/ which have .pt files
            with extracts from songs
    Return:
        a list of SI-SDR scores over 1s for vocals for the given type, a list of SDR
        scores for vocals, a list of SI-SDR scores for accompaniment, a list of SDR
        scores for accompaniment, a list of sub-track names (the file names in the
        /type/ folder)
    """
    
    root = args.root
    
    # to modify depending on your case
    lyrics_folder = '/tsi/doctorants/kschulze/Datasets/MUSDB_w_lyrics/lyrics_transcripts/test'
    
    # output lists
    SI_SDRscores_vocals = []
    SDRscores_vocals = []
    SI_SDRscores_accompaniment = []
    SDRscores_accompaniment = []
    
    if not os.path.exists(args.evaldir):
        os.makedirs(args.evaldir)
    
    for filename in tqdm.tqdm(os.listdir(lyrics_folder)):
        if filename.endswith('.txt'):
            # make file name for audio and text files of current track
            song_name_short = filename.replace('.txt','')
            song_name_short = song_name_short.split('-')
            song_name_short = song_name_short[0][0:6] + "_" + song_name_short[1][1:6]
            song_name_short = song_name_short.replace(" ", "_")
            
            # load the posteriogram corresponding to the current track (to be
            # sliced later to have just the piece corresponding to the extract)
            phoneme = torch.load(args.root_phoneme+'/'
                                    +'test'+'_'+filename.replace('.txt','.pt'))
            
            with open(lyrics_folder+'/'+filename) as f:
                for i,line in enumerate(f):
                    line = line.rstrip()
                    if line.startswith('*'):
                        continue
                    line = line.split(' ')
                    if line[2] != typee:
                        continue
                    
                    # extract start and end times an extract of type typee
                    # in the current song
                    start = get_sec(line[0])
                    end = get_sec(line[1])
                    
                    # this is NOT the real start frame, but the last to one.
                    # This is for conserving the behavior as in training where
                    # we pad one frame with 0, see test.py
                    startFrame = math.floor(start/0.016)
                    
                    # last overlaping frame
                    endFrame = math.ceil(end/0.016) - 1
                    sub_phoneme = phoneme[startFrame:endFrame+1]
                    
                    # to adjust padding at start
                    offset = start - math.floor(start/0.016)*0.016
                    
                    # load the audio from the /typee/ folder
                    path = root+'/mix/'+typee+'/'+song_name_short+'_'+str(i)+'.pt'
                    mixture = torch.load(path)
                    mixture = torch.cat([torch.zeros(2,int(offset*16000)),mixture],
                                        dim=1)
                    
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
                        enforce_fake=args.enforce_fake
                    )
                    
                    # write the estimate for the extract
                    if args.outdir:
                        if not os.path.exists(args.outdir):
                            os.makedirs(args.outdir)
                        if not os.path.exists(args.outdir+'/'+typee):
                            os.makedirs(args.outdir+'/'+typee)
                        
                        for target, estimate in list(estimates.items()):
                            target_path = (args.outdir+'/'+typee
                                            +'/'+song_name_short+str(i)
                                            +'_'+target+'.wav')
                            sf.write(target_path, estimate, 16000)

                    # we pad vocals with 0 to fit estimate shape, as we padded it
                    # before to have it start at the same time as a phoneme window
                    vocals = torch.load(root+'/vocals/'+typee+
                                        '/'+song_name_short+'_'+str(i)+'.pt')
                    vocals = torch.cat(
                                [torch.zeros(2,int(offset*16000)),vocals],
                                dim=1)
                    
                    estimated_vocals = torch.from_numpy(estimates['vocals'].T)
                    
                    if vocals.shape != estimated_vocals.shape:
                        raise ValueError("Targets and estimates should"+
                                        " have the same shape!")

                    vocals_SISDR = sisdr_framewise(estimated_vocals,vocals,
                                                    sample_rate=16000,eps=0)
                    vocals_SDR = ideal_SDR_framewise(estimated_vocals,
                                                    vocals,sample_rate=16000)
                    # mean over channels
                    vocals_SISDR = torch.mean(vocals_SISDR,dim=1)[0]
                    vocals_SDR = torch.mean(vocals_SDR,dim=1)[0]

                    accompaniment = torch.load(root+'/accompaniments/'+typee+
                                                '/'+song_name_short+'_'+str(i)+'.pt')
                    accompaniment = torch.cat(
                                [torch.zeros(2,int(offset*16000)),accompaniment],
                                dim=1)
                    estimated_accompaniment = torch.from_numpy(
                                                estimates['accompaniment'].T)
                    accompaniment_SISDR  = sisdr_framewise(
                                                estimated_accompaniment,
                                                accompaniment,
                                                sample_rate=16000,
                                                eps=0)
                    accompaniment_SDR  = ideal_SDR_framewise(
                                                estimated_accompaniment,
                                                accompaniment,
                                                sample_rate=16000)
                    accompaniment_SISDR  = torch.mean(accompaniment_SISDR,dim=1)[0]
                    accompaniment_SDR  = torch.mean(accompaniment_SDR,dim=1)[0]


                    frame_list = []
                    
                    # fill a list over each seconds with the metrics, that will
                    #be dumped to a .json file
                    for i,k in enumerate(vocals_SISDR):
                        frame_list.append({
                            "time" : float(i),
                            "duration" : 1.0,
                            "metrics" : {
                            "SI-SDR_vocals" : -vocals_SISDR[i].item(),
                            "SI-SDR_accompaniment" : -accompaniment_SISDR[i].item(),
                            "SDR_vocals" : -vocals_SDR[i].item(),
                            "SDR_accompaniment" : -accompaniment_SDR[i].item()
                        }})
                    
                    # median over windows
                    vocals_SISDR = vocals_SISDR[torch.isfinite(vocals_SISDR)]
                    vocals_SDR = vocals_SDR[torch.isfinite(vocals_SDR)]

                    median_SISDR_vocals = np.median(vocals_SISDR.numpy())
                    median_SDR_vocals = np.median(vocals_SDR.numpy())

                    SI_SDRscores_vocals.append(-median_SISDR_vocals)
                    SDRscores_vocals.append(-median_SDR_vocals)
                    
                    
                    medians = [{"median_SISDR_vocals" : -median_SISDR_vocals.item(),
                                "median_SDR_vocals" : -median_SDR_vocals.item()}]
                    
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
                                                - median_SISDR_accompaniment.item())
                    medians[0]["median_SDR_accompaniment"] = (
                                                -median_SDR_accompaniment.item())
                    
                    directory = args.evaldir+'/'+typee+'/'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    
                    with open(directory+song_name_short+str(i)+'.json', 'w') as outfile:
                        todump = [medians,frame_list]
                        json.dump(todump, outfile, indent=2)
                    
                    torch.cuda.empty_cache()
            
    return (SI_SDRscores_vocals,SDRscores_vocals,SI_SDRscores_accompaniment,
            SDRscores_accompaniment)


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
        default='/tsi/doctorants/kschulze/Datasets/MUSDB_w_lyrics/test/audio',
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
    
    seed_all(results['args']['seed'])
    
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
    
    for typee in tqdm.tqdm(types):
        (SI_SDRscores_vocals[typee],SDRscores_vocals[typee],
        SI_SDRscores_accompaniment[typee],
        SDRscores_accompaniment[typee]) = evalTargets(args,device,typee)
        
        
        overall_SI_SDR_vocals = np.median(np.array(SI_SDRscores_vocals[typee]))
        overall_SDR_vocals = np.median(np.array(SDRscores_vocals[typee]))
        
        text_file = open(args.evaldir+"/_overall.txt", "a")
        text_file.write("("+typee+") SI-SDR for vocals (median over windows, median over files): "
                        + str(overall_SI_SDR_vocals))
        text_file.write("\n")
        text_file.write("("+typee+") SDR for vocals (median over windows, median over files): "
                        + str(overall_SDR_vocals))
        
        
        overall_SI_SDR_accompaniment = np.median(np.array(SI_SDRscores_accompaniment[typee]))
        overall_SDR_accompaniment = np.median(np.array(SDRscores_accompaniment[typee]))
        
        text_file.write("\n-\n")
        text_file.write("("+typee+") SI-SDR for accompaniment (median over windows,"
                        +" median over files): " + str(overall_SI_SDR_accompaniment))
        text_file.write("\n")
        text_file.write("("+typee+") SDR for accompaniment (median over windows,"
                        " median over files): " + str(overall_SDR_accompaniment))
        
        text_file.write("\n\n")
        text_file.write("---------")
        text_file.write("\n\n")
        
    text_file.close()