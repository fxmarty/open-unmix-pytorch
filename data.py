from utils import load_audio, load_info
from pathlib import Path
import torch.utils.data
import argparse
import random
import musdb
import torch
import tqdm
import librosa

import random

class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

def _augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain between `low` and `high`"""
    g = low + torch.rand(1) * (high - low)
    return audio * g

def _augment_channelswap(audio):
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.FloatTensor(1).uniform_() < 0.5:
        return torch.flip(audio, [0])
    else:
        return audio

def load_datasets(parser, args):
    """Loads the specified dataset from commandline arguments
    
    Returns:
        train_dataset, validation_dataset
    """
    if args.dataset == 'musdb': # Default case
        parser.add_argument('--is-wav','--is_wav',
                            action='store_true', default=False,
                            help='loads wav instead of STEMS')
        parser.add_argument('--samples-per-track','--samples_per_track',
                            type=int, default=64)
        parser.add_argument('--source-augmentations','--source_augmentations',
                            type=str, nargs='+', default=['gain', 'channelswap'])

        args = parser.parse_args()
        dataset_kwargs = {
            'root': args.root,
            'is_wav': args.is_wav,
            'subsets': 'train',
            'target': args.target,
            'download': args.root is None,
            'random_chunks': args.random_chunks
        }

        source_augmentations = Compose(
            [globals()['_augment_' + aug] for aug in args.source_augmentations]
        )
        
        train_dataset = MUSDBDataset(
            modelname = args.modelname,
            split='train',
            samples_per_track=args.samples_per_track,
            seq_duration=args.seq_dur,
            source_augmentations=source_augmentations,
            random_track_mix=True,
            data_augmentation=args.data_augmentation,
            nb_channels=args.nb_channels,
            joint=args.joint,
            **dataset_kwargs
        )

        # samples_per_track is 1 for stereo case, 2 for mono case, and that is
        # because MUSDB is a stereo dataset. If we train for the mono case,
        # both left and right tracks will be used for validation.
        valid_dataset = MUSDBDataset(
            modelname = args.modelname, split='valid',
            samples_per_track=3-args.nb_channels, seq_duration=None,
            nb_channels=args.nb_channels,joint=args.joint,
            **dataset_kwargs
        )

    return train_dataset, valid_dataset, args

class MUSDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        modelname,
        joint,
        target='vocals',
        root=None,
        download=False,
        is_wav=False,
        subsets='train',
        split='train',
        seq_duration=6.0,
        samples_per_track=64,
        source_augmentations=lambda audio: audio,
        random_track_mix=False,
        dtype=torch.float32,
        data_augmentation="yes",
        random_chunks=True,
        nb_channels=2,
        *args, **kwargs
    ):
        """MUSDB18 torch.data.Dataset that samples from the MUSDB tracks
        using track and excerpts with replacement.

        Parameters
        ----------
        target : str
            target name of the source to be separated, defaults to ``vocals``.
        root : str
            root path of MUSDB
        download : boolean
            automatically download 7s preview version of MUSDB
        is_wav : boolean
            specify if the WAV version (instead of the MP4 STEMS) are used
        subsets : list-like [str]
            subset str or list of subset. Defaults to ``train``.
        split : str
            use (stratified) track splits for validation split (``valid``),
            defaults to ``train``.
        seq_duration : float
            training is performed in chunks of ``seq_duration`` (in seconds,
            defaults to ``None`` which loads the full audio track
        samples_per_track : int
            sets the number of samples, yielded from each track per epoch.
            Defaults to 64
        source_augmentations : list[callables]
            provide list of augmentation function that take a multi-channel
            audio file of shape (src, samples) as input and output. Defaults to
            no-augmentations (input = output)
        random_track_mix : boolean
            randomly mixes sources from different tracks to assemble a
            custom mix. This augmenation is only applied for the train subset.
        random_chunks : boolean
            boolean defining if chunks for the tracks will be choosed randomly or not
        dtype : numeric type
            data type of torch output tuple x and y
        args, kwargs : additional keyword arguments
            used to add further control for the musdb dataset
            initialization function.

        """
        self.modelname = modelname
        self.joint = joint
        self.is_wav = is_wav
        self.seq_duration = seq_duration
        self.target = target
        self.subsets = subsets
        self.split = split
        self.samples_per_track = samples_per_track
        self.source_augmentations = source_augmentations
        self.random_track_mix = random_track_mix
        self.data_augmentation = data_augmentation
        self.random_chunks = random_chunks
        self.nb_channels = nb_channels
        self.mus = musdb.DB(
            root=root,
            is_wav=is_wav,
            split=split,
            subsets=subsets,
            download=download,
            *args, **kwargs
        )
        
        if len(self.mus.tracks) > 0:
            self.sample_rate = self.mus.tracks[0].rate
        
        else:
            self.sample_rate = 8192 # to modify manually for minimal tests
        
        self.dtype = dtype
        
        if self.data_augmentation == 'no': # save tracks numbers
            self.dataindex = torch.zeros(len(self.mus),self.samples_per_track)
            for i in range(len(self.mus)):
                track = self.mus.tracks[i]
                for j in range(self.samples_per_track):
                    chunk_start = random.uniform(0,
                                            track.duration - self.seq_duration)
                    self.dataindex[i][j] = chunk_start
                    
    def __getitem__(self, index):
        audio_sources = []
        target_ind = None
        """
        if torch.utils.data.get_worker_info() is not None:
            print(torch.utils.data.get_worker_info())
        print(random.random())
        print("---")
        """

        # select track
        track = self.mus.tracks[index // self.samples_per_track]

        # at training time we assemble a custom mix
        if self.split == 'train' and self.seq_duration:
            for k, source in enumerate(self.mus.setup['sources']):
                # memorize index of target source
                if source == self.target:
                    target_ind = k
                
                # select a random track if data augmentation
                if self.random_track_mix and self.data_augmentation == 'yes':
                    track = random.choice(self.mus.tracks)
                
                # set the excerpt duration
                track.chunk_duration = self.seq_duration
                
                # set random start position if data augmentation
                if self.random_chunks:
                    track.chunk_start = random.uniform(
                        0, track.duration - self.seq_duration)
                else:
                    track.chunk_start = self.dataindex[index // self.samples_per_track][index % self.samples_per_track].item()
                
                # load source audio and apply time domain source_augmentations
                audio = torch.tensor(
                    track.sources[source].audio.T,
                    dtype=self.dtype
                )
                
                if self.modelname == 'deep-u-net':
                    audio = audio[...,:98560] # for testing purpose, 128 frames
                                
                if self.data_augmentation == 'yes':
                    audio = self.source_augmentations(audio)
                
                if self.nb_channels == 1: # select randomly left or right channel
                    channel_number = 0
                    if self.random_chunks: channel_number = random.randint(0, 1) 
                    audio = torch.unsqueeze(audio[channel_number],0)
                
                audio_sources.append(audio)
                    
            # create stem tensor of shape (nb_sources=4, nb_channels, samples)
            stems = torch.stack(audio_sources, dim=0)
            # apply linear mix over source index=0
            x = stems.sum(0)
            # get the target stem
            if target_ind is not None:
                y = stems[target_ind]
            else:
                raise ValueError("Target has not been given.")
            
            if self.modelname == 'convtasnet':
                if self.joint == True:
                    y_accompaniment = x - y
                    y = torch.stack((y,y_accompaniment),dim=0)
                else:
                    y = torch.unsqueeze(y,0)
        # for validation and test, we deterministically yield the full
        # pre-mixed musdb track
        else:
            # get the non-linear source mix straight from musdb
            x = torch.tensor(
                track.audio.T,
                dtype=self.dtype
            )
            y = torch.tensor(
                track.targets[self.target].audio.T,
                dtype=self.dtype
            )
            if self.nb_channels == 1: # select left or right depending on index even or not
                x = torch.unsqueeze(x[index%2],0)
                y = torch.unsqueeze(y[index%2],0)
            # if nb_channels = 2, use both channels
            
            if self.modelname == 'convtasnet':
                if self.joint == True:
                    y_accompaniment = x - y
                    y = torch.stack((y,y_accompaniment),dim=0)
                else:
                    y = torch.unsqueeze(y,0)
            
            #x = x[...,:x.shape[-1]//2]
            #y = y[...,:y.shape[-1]//2]

        return x, y

    def __len__(self):
        # self.mus.tracks : liste of the names of the tracks in train folder
        # self.samples_per_track : parameter, default 64
        return len(self.mus.tracks) * self.samples_per_track
