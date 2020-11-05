import torch
import torch.nn as nn
import numpy as np

import argparse
import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

import data
import utils_encoder
import model

from datetime import datetime
import time
from pathlib import Path
import sys
import json
import copy


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

def train(args, autoencoder, device, train_sampler, optimizer):
    losses = utils_encoder.AverageMeter()
    autoencoder.train()
    pbar = tqdm.tqdm(train_sampler)
    i = 0
    # Loop by number of tracks * number of samples per track / batch size
    for phoneme in pbar:
        pbar.set_description("Training batch")
        phoneme = phoneme.to(device)
        optimizer.zero_grad()
        
        sys.stdout.write("\rBatch number %i" % i)
        sys.stdout.flush()
        
        estimate = autoencoder(phoneme)

        loss = torch.nn.functional.mse_loss(estimate, phoneme)# + phoneme
        
        # estimate.size(0) for batch size
        losses.update(loss.item(), estimate.size(0))
        
        i = i + 1

        loss.backward()
        optimizer.step()
    
    return losses.avg

def valid(args, autoencoder, device, valid_sampler):
    losses = utils_encoder.AverageMeter()
    autoencoder.eval()
    with torch.no_grad():
        for phoneme in valid_sampler:
            phoneme = phoneme.to(device)
            estimate = autoencoder(phoneme)
            
            loss = torch.nn.functional.mse_loss(estimate, phoneme)
            
            losses.update(loss.item(), estimate.size(0))
        
        return losses.avg

def main():    
    parser = argparse.ArgumentParser(description='Phoneme auto-encoder')
        
    parser.add_argument('--root-phoneme', type=str, required=True,
                        help='root path of .pt phonemes')
    
    parser.add_argument('--output', type=str,required=True,
                        help='provide output path base folder name')
    
    parser.add_argument('--model', type=str, help='Path to checkpoint folder')

    # Training Parameters
    parser.add_argument('--epochs', type=int, default=1000)
    
    parser.add_argument('--batch-size','--batch_size', type=int, default=16)

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, defaults to 1e-3')
    
    parser.add_argument('--patience', type=int, default=20,
                        help='maximum number of epochs to train without improvement of the validation loss(default: 140)')
    
    parser.add_argument('--lr-decay-patience','--lr_decay_patience',type=int,
                        default=10, help='lr decay patience for plateau scheduler')
                        
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
        
    parser.add_argument('--nb-workers','--nb_workers', type=int, default=0,
                        help='Number of workers for dataloader.')

    # Misc Parameters    
    parser.add_argument('--no-cuda','--no_cuda', action='store_true',
                        default=False, help='disables CUDA training')
            
    parser.add_argument('--tb', default=None,
                        help='use tensorboard, and if so, set name')
    
    parser.add_argument('--bottleneck-size', default=8,
                        help='Set the bottleneck size')
        
    args, _ = parser.parse_known_args()
    
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
    
    dataloader_kwargs = {'num_workers': args.nb_workers, 
                        'pin_memory': True} if use_cuda else {}

    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Load and process training and validation sets
    train_dataset, valid_dataset, args = data.load_datasets(parser, args)
    
    # create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)
    
    print("Number of batches per epoch:",len(train_dataset)/args.batch_size)
    
    number_of_phonemes = train_dataset[0].shape[1]
    
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
    
        
    autoencoder = model.Autoencoder(
        number_of_phonemes=number_of_phonemes,
        bottleneck_size=args.bottleneck_size
    ).to(device)

    optimizer = torch.optim.Adam(
        autoencoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10
    )

    es = utils_encoder.EarlyStopping(patience=args.patience)
    
    # Use tensorboard if specified as an argument
    if args.tb is not None:
        currentDay = datetime.now().strftime('%m-%d_')
        currentHour = datetime.now().strftime('_%H:%M_')
        global writerTrainLoss
        global writerValidationLoss
        trainDirName = "runs-autoencoder/" + currentDay + args.tb + currentHour + "train"
        validDirName = "runs-autoencoder/" + currentDay + args.tb + currentHour + "validation"
        
        writerTrainLoss = SummaryWriter(log_dir=trainDirName)
        writerValidationLoss = SummaryWriter(log_dir=validDirName)
        
        argsCopyDict = vars(copy.deepcopy(args))
        del argsCopyDict['output']
        del argsCopyDict['model']
        del argsCopyDict['seed']
        del argsCopyDict['nb_workers']
        del argsCopyDict['no_cuda']
        writerTrainLoss.add_hparams(argsCopyDict,{'hparam/losss': 1})
    
    # if a model is specified: resume training
    if args.model:
        model_path = Path(args.model).expanduser()
        with open(Path(model_path, 'model.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, "model.chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        autoencoder.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # train for another epochs_trained
        t = tqdm.trange(
            results['epochs_trained'],
            results['epochs_trained'] + args.epochs + 1
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
        t = tqdm.trange(1, args.epochs + 1)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()
        
        for param_group in optimizer.param_groups:
            print("Learning rate:", param_group['lr'])

        train_loss = train(args, autoencoder, device, train_sampler, optimizer)
        valid_loss = valid(args, autoencoder, device, valid_sampler)

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
        
        print("Epoch ",epoch,", train loss: ",train_loss, "valid loss: ",valid_loss)
        
        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch
        
        utils_encoder.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': autoencoder.state_dict(),
                'encoder_state_dict': autoencoder.encoder.state_dict(),
                'best_loss': es.best,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
            is_best=valid_loss == es.best,
            path=target_path
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
            'num_bad_epochs': es.num_bad_epochs
        }

        with open(Path(target_path,  'model.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))
        
        if stop:
            print("Apply Early Stopping")
            break
    
    if args.tb is not None:
        writerTrainLoss.close()
        writerValidationLoss.close()

if __name__ == "__main__":
    main()

