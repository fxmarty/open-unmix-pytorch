import argparse
import model
import data
import torch
import time
from pathlib import Path
import tqdm
import json
import utils
import sklearn.preprocessing
import numpy as np
import random
from git import Repo
import os
import copy
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import sys

currentTime = datetime.now().strftime('%m-%d_%H:%M')
writerTrainLoss = SummaryWriter(log_dir="runs/" + currentTime + "-train")
writerValidationLoss = SummaryWriter(log_dir="runs/" + currentTime + "-validation")

tqdm.monitor_interval = 0
batch_seen = 0


def train(args, unmix, device, train_sampler, optimizer):
    losses = utils.AverageMeter()
    unmix.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    global batch_seen
    # Loop by number of tracks * number of samples per track / batch size
    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        Y_hat = unmix(x)
        Y = unmix.transform(y)
        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        loss.backward()
        optimizer.step()
        
        batch_seen = batch_seen + 1 
        writerTrainLoss.add_scalar('Loss', loss.item(),batch_seen)
        
        losses.update(loss.item(), Y.size(1))
    return losses.avg


def valid(args, unmix, device, valid_sampler):
    losses = utils.AverageMeter()
    unmix.eval()
    global batch_seen
    with torch.no_grad():
        for x, y in valid_sampler:
            x, y = x.to(device), y.to(device)
            Y_hat = unmix(x)
            Y = unmix.transform(y)
            loss = torch.nn.functional.mse_loss(Y_hat, Y)
            losses.update(loss.item(), Y.size(1))
        
        print("Valid:",losses.avg)
        writerValidationLoss.add_scalar('Loss', losses.avg,batch_seen)
        
        return losses.avg


def get_statistics(args, dataset):
    scaler = sklearn.preprocessing.StandardScaler()

    spec = torch.nn.Sequential(
        model.STFT(n_fft=args.nfft, n_hop=args.nhop),
        model.Spectrogram(mono=True)
    )

    dataset_scaler = copy.deepcopy(dataset)
    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_chunks = False
    dataset_scaler.random_track_mix = False
    dataset_scaler.random_interferer_mix = False
    dataset_scaler.seq_duration = None
    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=args.quiet)
    for ind in pbar:
        x, y = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        X = spec(x[None, ...])
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

    # Dataset paramaters
    parser.add_argument('--dataset', type=str, default="musdb",
                        choices=[
                            'musdb', 'aligned', 'sourcefolder',
                            'trackfolder_var', 'trackfolder_fix'
                        ],
                        help='Name of the dataset.')
    parser.add_argument('--root', type=str, help='root path of dataset')
    parser.add_argument('--output', type=str, default="open-unmix",
                        help='provide output path base folder name')
    parser.add_argument('--model', type=str, help='Path to checkpoint folder')

    # Trainig Parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, defaults to 1e-3')
    parser.add_argument('--patience', type=int, default=140,
                        help='maximum number of epochs to train (default: 140)')
    parser.add_argument('--lr-decay-patience', type=int, default=80,
                        help='lr decay patience for plateau scheduler')
    parser.add_argument('--lr-decay-gamma', type=float, default=0.3,
                        help='gamma of learning rate scheduler decay')
    parser.add_argument('--weight-decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    # Model Parameters
    parser.add_argument('--seq-dur', type=float, default=6.0,
                        help='Sequence duration in seconds'
                        'value of <=0.0 will use full/variable length')
    parser.add_argument('--unidirectional', action='store_true', default=False,
                        help='Use unidirectional LSTM instead of bidirectional')
    parser.add_argument('--nfft', type=int, default=4096,
                        help='STFT fft size and window size')
    parser.add_argument('--nhop', type=int, default=1024,
                        help='STFT hop size')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='hidden size parameter of dense bottleneck layers')
    parser.add_argument('--bandwidth', type=int, default=16000,
                        help='maximum model bandwidth in herz')
    parser.add_argument('--nb-channels', type=int, default=2,
                        help='set number of channels for model (1, 2)')
    parser.add_argument('--nb-workers', type=int, default=0,
                        help='Number of workers for dataloader.')

    # Misc Parameters
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='less verbose during training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args, _ = parser.parse_known_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    print("Using Torchaudio: ", utils._torchaudio_available())
    dataloader_kwargs = {'num_workers': args.nb_workers, 'pin_memory': True} if use_cuda else {}

    repo_dir = os.path.abspath(os.path.dirname(__file__))
    repo = Repo(repo_dir)
    commit = repo.head.commit.hexsha[:7]

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset, valid_dataset, args = data.load_datasets(parser, args)
    print("Taille validation set:",len(valid_dataset))
    print("Taille train set:",len(train_dataset))
    # When working with MUSDB, train_dataset of type MUSDBDataset
    # Pretty much a dataset of size samples_per_track * number of tracks,
    # With each having both the mixture and the target source in stereo.
    
    # By definition of __getitem__ for MUSDBDataset in data.py, this is actually
    # a random result at each call of __getitem__!
    
    # Returns torch.Size([2, 264600]) by default (2 for stereo, 264600 = 6*44100)
    # In train_dataset[0][0], first 0 for number of sample, second 0 for the mixture
    # (may be 1 for the target source)
    #print("1er élément:",train_dataset[0][0].shape)
    
    # create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        **dataloader_kwargs
    )

    valid_sampler = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1,
        **dataloader_kwargs
    )

    if args.model:
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics(args, train_dataset)

    max_bin = utils.bandwidth_to_max_bin(
        train_dataset.sample_rate, args.nfft, args.bandwidth
    ) # to stay under 16 000 Hz

    unmix = model.OpenUnmix(
        input_mean=scaler_mean,
        input_scale=scaler_std,
        nb_channels=args.nb_channels,
        hidden_size=args.hidden_size,
        n_fft=args.nfft,
        n_hop=args.nhop,
        max_bin=max_bin,
        sample_rate=train_dataset.sample_rate
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
        train_loss = train(args, unmix, device, train_sampler, optimizer)
        valid_loss = valid(args, unmix, device, valid_sampler)
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(
            train_loss=train_loss, val_loss=valid_loss
        )

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

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break

writerTrainLoss.close()
writerValidationLoss.close()

if __name__ == "__main__":
    main()
