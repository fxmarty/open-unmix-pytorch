import argparse

from models import open_unmix

import data
import utils
import copy

import torch
import torch.nn as nn
import torchaudio
torchaudio.set_audio_backend("soundfile")

import time
from pathlib import Path
import tqdm
import json
import sklearn.preprocessing
import numpy as np
import random
from git import Repo
import os
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from datetime import datetime
import sys
import normalization

from SDR_metrics import sisdr_framewise
from SDR_metrics import loss_SI_SDR

from utils import memory_check
import tf_transforms

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D

import torchsnooper
import time

# Overright SummaryWriter so that hparams is in the same subfolder as the rest
class SummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)

batch_seen = 0

#@torchsnooper.snoop()
def train(args, unmix, device, train_sampler, optimizer,model_name_general,epoch_num,tb="no"):
    losses = utils.AverageMeter()
    unmix.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    global batch_seen
    i = 0
    # Loop by number of tracks * number of samples per track / batch size
    for x, phoneme, y in pbar:
        pbar.set_description("Training batch")
        x, phoneme, y = x.to(device), phoneme.to(device), y.to(device)
        optimizer.zero_grad()
        
        sys.stdout.write("\rBatch number %i" % i)
        sys.stdout.flush()
        if model_name_general in ('open-unmix'):
            Y_hat = unmix(x,phoneme)
            Y = unmix.transform(y)

            loss = torch.nn.functional.mse_loss(Y_hat, Y)
            
            losses.update(loss.item(), Y.size(1))
        
        i = i + 1

        loss.backward()
        optimizer.step()

        del Y_hat
        del x
        del y
        del phoneme
        torch.cuda.empty_cache()
    
    return losses.avg


def valid(args, unmix, device, valid_sampler,model_name_general,tb="no"):
    losses = utils.AverageMeter()
    unmix.eval()
    global batch_seen
    with torch.no_grad():
        for x, phoneme, y in valid_sampler:
            x, phoneme, y = x.to(device), phoneme.to(device), y.to(device)
            if model_name_general in ('open-unmix'):
                Y_hat = unmix(x,phoneme)
                Y = unmix.transform(y)
                
                loss = torch.nn.functional.mse_loss(Y_hat, Y)
                
                losses.update(loss.item(), Y.size(1))
            
            del Y_hat
            del x
            del y
            torch.cuda.empty_cache()

        return losses.avg

# Called only when normalization_style = overall
def get_statistics(args, dataset):
    scaler = sklearn.preprocessing.StandardScaler()

    spec = torch.nn.Sequential(
        tf_transforms.STFT(n_fft=args.nfft, n_hop=args.nhop),
        tf_transforms.Spectrogram()
    )

    dataset_scaler = copy.deepcopy(dataset)
    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_track_mix = False
    dataset_scaler.random_interferer_mix = False
    dataset_scaler.seq_duration = None
    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=args.quiet)
    for ind in pbar:
        x, phoneme, y = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        X = spec(x[None, ...])
        if args.nb_channels == 2: # required by partial_fit
            X = torch.mean(X, 2, keepdim=True)
        scaler.partial_fit(np.squeeze(X))

    # set inital input scaler values
    std = np.maximum(
        scaler.scale_,
        1e-4*np.max(scaler.scale_)
    )
    return scaler.mean_, std

def main():    
    parser = argparse.ArgumentParser(description='Open Unmix Trainer')

    # which target do we want to train?
    parser.add_argument('--target', type=str, default='vocals',
                        help='target source (will be passed to the dataset)')
    
    parser.add_argument('--root', type=str, help='root path of dataset')
    
    # BEWARE: at the STFT time resolution
    parser.add_argument('--root-phoneme', type=str, help='root path of .pt phonemes')
    
    parser.add_argument('--output', type=str, default="open-unmix",
                        help='provide output path base folder name')
    
    parser.add_argument('--model', type=str, help='Path to checkpoint folder')

    # Trainig Parameters
    parser.add_argument('--epochs', type=int, default=1000)
    
    parser.add_argument('--batch-size','--batch_size', type=int, default=16)

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, defaults to 1e-3')
    
    parser.add_argument('--patience', type=int, default=140,
                        help='maximum number of epochs to train without improvement of the validation loss(default: 140)')
    
    parser.add_argument('--lr-decay-patience','--lr_decay_patience',type=int,
                        default=80, help='lr decay patience for plateau scheduler')
                        
    parser.add_argument('--lr-decay-gamma','--lr_decay_gamma', type=float,
                        default=0.3, help='gamma of learning rate scheduler decay')
    parser.add_argument('--weight-decay','--weight_decay', type=float,
                        default=0.00001,help='weight decay')
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')

    # Model Parameters
    parser.add_argument('--seq-dur','--seq_dur', type=float, default=6.0,
                        help='Sequence duration in seconds'
                        'value of <=0.0 will use full/variable length')
    
    parser.add_argument('--unidirectional', action='store_true', default=False,
                        help='Use unidirectional LSTM instead of bidirectional')
    
    # corresponds to 0.032 s at 16 kHz
    parser.add_argument('--nfft', type=int, default=512,
                        help='STFT fft size and window size')
    
    # corresponds to 0.016 s at 16 kHz
    parser.add_argument('--nhop', type=int, default=256,
                        help='STFT hop size')
    
    parser.add_argument('--hidden-size','--hidden_size', type=int, default=512,
                        help='hidden size parameter of dense bottleneck layers')
    
    parser.add_argument('--nb-channels','--nb_channels', type=int, default=2,
                        help='set number of channels for model (1, 2)')
    
    parser.add_argument('--nb-workers','--nb_workers', type=int, default=0,
                        help='Number of workers for dataloader.')

    # Misc Parameters
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='less verbose during training')
    
    parser.add_argument('--no-cuda','--no_cuda', action='store_true',
                        default=False, help='disables CUDA training')
    
    parser.add_argument(
        '--modelname',
        choices=['open-unmix'],
        type=str,
        help='model name, used to modify the training procedure accordingly'
    )
    
    parser.add_argument(
        '--normalization-style','--normalization_style',
        choices=['overall', 'batch-specific','none'],
        type=str,
        help='Use different normalization styles than the default for a model.'
    )
    
    parser.add_argument('--tb', default=None,
                        help='use tensorboard, and if so, set name')

    parser.add_argument('--fake',
                        action='store_true',
                        help='Input fake constant phoneme')
    
    parser.add_argument('--no-random-track-mix',
                        action='store_false',
                        help='Disable random track selection for each target in the dataset')
    
    parser.add_argument('--no-random-chunk-start',
                        action='store_false',
                        help='Disable random chunk start for each target in the dataset')

    parser.add_argument('--no-source-augmentation',
                        action='store_false',
                        help='Disable source augmentation for each target in the dataset')

    parser.add_argument('--no-random-channel',
                        action='store_false',
                        help='Disable random channel selection for each target in the dataset')
    
    args, _ = parser.parse_known_args()
    
    # Make normalization-style argument not mendatory
    if args.normalization_style == None and args.modelname == 'open-unmix':
        parser.set_defaults(normalization_style='overall')
        
    # Update args according to the conditions above
    args, _ = parser.parse_known_args()
    
    # enforce deterministic behavior
    def seed_all(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    seed_all(args.seed)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    print("Using Torchaudio: ", utils._torchaudio_available())
    
    dataloader_kwargs = {'num_workers': args.nb_workers, 
                        'pin_memory': True} if use_cuda else {}

    repo_dir = os.path.abspath(os.path.dirname(__file__))
    repo = Repo(repo_dir)
    commit = repo.head.commit.hexsha[:7]

    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Load and process training and validation sets
    train_dataset, valid_dataset, args = data.load_datasets(parser, args)
    
    # create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)
    
    print("Sampling rate of dataset:",train_dataset.sample_rate,"Hz")
    print("Size validation set:",len(valid_dataset),"(",len(valid_dataset.mus.tracks),
            "*",valid_dataset.samples_per_track,", number of tracks * samples per track)")
    print("Size train set:",len(train_dataset),"(",len(train_dataset.mus.tracks),
            "*",train_dataset.samples_per_track,", number of tracks * samples per track)")
    print("Number of batches per epoch:",len(train_dataset)/args.batch_size)
    
    number_of_phonemes = train_dataset[0][1].shape[1]
    
    # called at every epoch, where initial_seed is different at every epoch and for
    # every worker.
    def _init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)
    
    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        worker_init_fn=_init_fn,
        **dataloader_kwargs
    )

    valid_sampler = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1,
        worker_init_fn=_init_fn,
        **dataloader_kwargs
    )
    
    if args.model or args.normalization_style in ('batch-specific','none'):
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics(args, train_dataset)
        #scaler_mean, scaler_std = None,None
        
    if args.modelname == 'open-unmix':
        unmix = open_unmix.OpenUnmix(
            normalization_style=args.normalization_style,
            input_mean=scaler_mean,
            input_scale=scaler_std,
            nb_channels=args.nb_channels,
            hidden_size=args.hidden_size,
            n_fft=args.nfft,
            n_hop=args.nhop,
            sample_rate=train_dataset.sample_rate,
            number_of_phonemes=number_of_phonemes
        ).to(device)

    optimizer = torch.optim.Adam(
        unmix.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10
    )

    es = utils.EarlyStopping(patience=args.patience)
    
    # Use tensorboard if specified as an argument
    if args.tb is not None:
        currentDay = datetime.now().strftime('%m-%d_')
        currentHour = datetime.now().strftime('_%H:%M_')
        global writerTrainLoss
        global writerValidationLoss
        trainDirName = "runs-"+args.modelname+"/" + currentDay + args.tb + currentHour + "train"
        validDirName = "runs-"+args.modelname+"/" + currentDay + args.tb + currentHour + "validation"
        
        writerTrainLoss = SummaryWriter(log_dir=trainDirName)
        writerValidationLoss = SummaryWriter(log_dir=validDirName)
        
        argsCopyDict = vars(copy.deepcopy(args))
        del argsCopyDict['output']
        del argsCopyDict['model']
        del argsCopyDict['seed']
        del argsCopyDict['unidirectional']
        del argsCopyDict['nb_workers']
        del argsCopyDict['quiet']
        del argsCopyDict['no_cuda']
        del argsCopyDict['is_wav']
        del argsCopyDict['source_augmentations']
        writerTrainLoss.add_hparams(argsCopyDict,{'hparam/losss': 1})
    
    
    # if a model is specified: resume training
    if args.model:
        model_path = Path(args.model).expanduser()
        with open(Path(model_path, args.target + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.target + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        unmix.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # train for another epochs_trained
        t = tqdm.trange(
            results['epochs_trained'],
            results['epochs_trained'] + args.epochs + 1,
            disable=args.quiet
        )
        train_losses = results['train_loss_history']
        valid_losses = results['valid_loss_history']
        train_times = results['train_time_history']
        best_epoch = results['best_epoch']
        es.best = results['best_loss']
        es.num_bad_epochs = results['num_bad_epochs']
    
    # else start from 0
    else:
        print("Learning starts from 0")
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()
        
        for param_group in optimizer.param_groups:
            print("Learning rate:", param_group['lr'])

        train_loss = train(args, unmix, device, train_sampler, optimizer,model_name_general=args.modelname,epoch_num=epoch,tb=args.tb)
        valid_loss = valid(args, unmix, device, valid_sampler,model_name_general=args.modelname,tb=args.tb)

        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        if args.tb is not None:
            writerTrainLoss.add_scalar('Loss', train_loss,epoch)
            writerValidationLoss.add_scalar('Loss', valid_loss,epoch)
            writerTrainLoss.flush()
            writerValidationLoss.flush()
            
        t.set_postfix(
            train_loss=train_loss, val_loss=valid_loss
        )
        
        print("Epoch ",epoch,", train loss: ",train_loss)
        
        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch
        
        utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': unmix.state_dict(),
                'best_loss': es.best,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
            is_best=valid_loss == es.best,
            path=target_path,
            target=args.target
        )
        
        # save params
        params = {
            'epochs_trained': epoch,
            'args': vars(args),
            'best_loss': es.best,
            'best_epoch': best_epoch,
            'train_loss_history': train_losses,
            'valid_loss_history': valid_losses,
            'train_time_history': train_times,
            'num_bad_epochs': es.num_bad_epochs,
            'commit': commit
        }

        with open(Path(target_path,  args.target + '.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))
        
        
        if stop:
            print("Apply Early Stopping")
            break
    
    if args.tb is not None:
        writerTrainLoss.close()
        writerValidationLoss.close()

if __name__ == "__main__":
    main()
