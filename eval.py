import argparse
import musdb
import museval
import test
import multiprocessing
import functools
from pathlib import Path
import torch
torch.cuda.init() # to run memory-check when we want (even at the beginning)

import tqdm
from loss_SI_SDR import sisdr_framewise
import numpy as np

from utils import memory_check

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

    args, _ = parser.parse_known_args()
    args = test.inference_args(parser, args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    mus = musdb.DB(
        root=args.root,
        download=args.root is None,
        subsets=args.subset,
        is_wav=args.is_wav
    )
    
    SI_SDRscores_vocals = []
    SI_SDRscores_accompaniment = []
    text_file = open(args.evaldir+"metrics.txt", "w")
    
    for track in tqdm.tqdm(mus.tracks):
        #track = mus.tracks[41]
        print(track.name)
        estimates = test.separate(
            audio=track.audio, # shape [nb_time_points, 2]
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
        accompaniment = torch.from_numpy(track.targets['accompaniment'].audio.T)
        
        estimated_vocals = torch.from_numpy(estimates['vocals'].T)
        estimated_accompaniment = torch.from_numpy(estimates['accompaniment'].T)
        
        if vocals.shape != estimated_vocals.shape:
            raise ValueError("Targets and estimates should have the same shape!")
        
        vocals_SISDR = sisdr_framewise(estimated_vocals,vocals,
                                        sample_rate=track.rate,eps=0)
        
        accompaniment_SISDR  = sisdr_framewise(estimated_accompaniment,accompaniment, sample_rate=track.rate,eps=0)
        
        SI_SDRscores_vocals.append(-vocals_SISDR)
        SI_SDRscores_accompaniment.append(-accompaniment_SISDR)
        print("vocals:",-vocals_SISDR)
        print("accomp:",-accompaniment_SISDR)
        print("sum:",-vocals_SISDR-accompaniment_SISDR)
        
        del vocals, accompaniment, estimated_vocals, estimated_accompaniment
        torch.cuda.empty_cache()
    
    summed = np.array(SI_SDRscores_vocals) + np.array(SI_SDRscores_accompaniment)
    print(summed)
    print("mean loss:", np.mean(summed))

    text_file.write("SI-SDR for vocals: " + str(np.mean(SI_SDRscores_vocals)))
    text_file.write("\n")
    text_file.write("SI-SDR for accompaniment: "
                        + str(np.mean(SI_SDRscores_accompaniment)))
    text_file.close()