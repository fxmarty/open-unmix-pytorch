from pathlib import Path
import argparse
import random
import torch
import math
import os

import random

def load_datasets(parser, args):
    """Loads the specified dataset from commandline arguments
    
    Returns:
        train_dataset, validation_dataset
    """
    parser.add_argument('--samples-per-track','--samples_per_track',
                        type=int, default=64)

    args = parser.parse_args()
    
    train_dataset = PhonemeDataset(
        split='train',
        samples_per_track=args.samples_per_track,
        seq_duration=args.seq_dur,
        root_phoneme=args.root_phoneme
    )

    valid_dataset = PhonemeDataset(
        split='valid',
        samples_per_track=1,
        seq_duration=None,
        root_phoneme = args.root_phoneme
    )

    return train_dataset, valid_dataset, args

class PhonemeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_phoneme=None,
        target='vocals',
        subsets='train',
        split='train',
        seq_duration=6.0,
        samples_per_track=64,
        dtype=torch.float32
    ):

        self.phoneme_window = 0.032 # in seconds
        self.phoneme_hop = 0.016 # in seconds
        
        self.seq_duration = seq_duration
        self.subsets = subsets
        self.split = split
        self.samples_per_track = samples_per_track
        
        self.root_phoneme = root_phoneme
        
        self.dtype = dtype
        
        self.phonemes_dict = {}
        
        self.train_song_names = []
        self.validation_song_names = []
        
        validation_song_list = ['train_Actions - One Minute Smile',
                                'train_Clara Berry And Wooldog - Waltz For My Victims',
                                'train_Johnny Lokke - Promises & Lies',
                                'train_Patrick Talbot - A Reason To Leave',
                                'train_Triviul - Angelsaint',
                                'train_Alexander Ross - Goodbye Bolero',
                                'train_Fergessen - Nos Palpitants',
                                'train_Leaf - Summerghost',
                                'train_Skelpolu - Human Mistakes',
                                'train_Young Griffo - Pennies',
                                'train_ANiMAL - Rockshow',
                                'train_James May - On The Line',
                                'train_Meaxic - Take A Step',
                                'train_Traffic Experiment - Sirens'
                            ]
        
        for filename in os.listdir(self.root_phoneme):
            if filename.endswith('.pt') and filename.startswith('train'):
                if filename.replace('.pt','') not in validation_song_list:
                    self.train_song_names.append(filename.replace('.pt',''))
                else:
                    self.validation_song_names.append(filename.replace('.pt',''))
                phoneme = torch.load(self.root_phoneme+'/'
                                +filename).float()
                self.phonemes_dict[filename.replace('.pt','')] = phoneme

    def __getitem__(self, index):
        
        # select track
        track_name = self.train_song_names[index // self.samples_per_track]
        track_duration = ((self.phonemes_dict[track_name].shape[0] - 1) 
                                * self.phoneme_hop) + self.phoneme_window

        # at training time we select random sections of seq_duration duration
        if self.split == 'train' and self.seq_duration:
            track_chunk_start = math.floor(random.uniform(
                0, min(track_duration,600) - self.seq_duration - 0.016)
                / self.phoneme_hop) * self.phoneme_hop
            
            startFrame = math.floor(track_chunk_start/self.phoneme_hop)
            nbFrames = math.floor((self.seq_duration - 0.032)/0.016 + 1)
            
            # phoneme shape before slice [phoneme_time_dim,nb_phoneme]
            phoneme = self.phonemes_dict[track_name][startFrame:startFrame+nbFrames]
        
        # for validation and test, we yield the full phoneme
        else:
            track_chunk_duration = min(track_duration - 0.016,600)
            nbFrames = math.floor((track_chunk_duration - 0.032)/0.016 + 1)
            
            phoneme = self.phonemes_dict[track_name][:nbFrames]
            
        # x shape [nb_channels, nb_time_frames]
        return phoneme

    def __len__(self):
        # self.samples_per_track : parameter, default 64
        if self.split == 'train':
            return len(self.train_song_names) * self.samples_per_track
        elif self.split == 'valid':
            return len(self.validation_song_names)
        else:
            raise ValueError("Split should be train or valid.")

if __name__ == '__main__':
    train_dataset = PhonemeDataset(
        split='train',
        samples_per_track=64,
        seq_duration=6,
        root_phoneme='/tsi/doctorants/fmarty/Posteriograms/10-19_trueVocals_ctc_moved'
    )
    
    print("length",len(train_dataset))
    print(train_dataset[0].shape)
    print("nb songs train",len(train_dataset.train_song_names))
    print("nb songs valid",len(train_dataset.validation_song_names))