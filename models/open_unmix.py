from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_model_summary
from torchsummary import summary
import matplotlib.pyplot as plt

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import time_transform_posteriograms
import normalization
import tf_transforms

class PhonemeNetwork(nn.Module):
    def __init__(
        self,
        phoneme_hidden_size,
        number_of_phonemes,
        fft_window_duration,
        fft_hop_duration,
        center
    ):
        super(PhonemeNetwork, self).__init__()
        
        self.fc1Phoneme = nn.Sequential(
                            Linear(number_of_phonemes, phoneme_hidden_size),
                            nn.ReLU()
                            )
        
        self.lstmPhoneme = LSTM(
            input_size=phoneme_hidden_size,
            hidden_size=phoneme_hidden_size//2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        self.fft_window_duration = fft_window_duration
        self.fft_hop_duration = fft_hop_duration
        self.center = center
    
    def forward(self, phoneme,nb_frames,offset):
        """
        #out [nb_samples, nb_frames, nb_phonemes]
        phoneme = time_transform_posteriograms.open_unmix(
                            phoneme,nb_frames,0.016,
                            self.fft_window_duration,self.fft_hop_duration,
                            center=self.center,offset=offset)
        """
        
        nb_samples, nb_frames,nb_phonemes = phoneme.shape
        
        # out [nb_samples, nb_frames, phoneme_hidden_size]
        phoneme = self.fc1Phoneme(phoneme)
            
        # out [nb_samples, nb_frames, phoneme_hidden_size]
        phoneme = self.lstmPhoneme(phoneme)[0]
        
        # to adapt to open-unmix, reshape to
        # [nb_frames,nb_samples,phoneme_hidden_size]
        phoneme = phoneme.reshape(nb_frames,nb_samples,-1)
        
        return phoneme

class OpenUnmix(nn.Module):
    def __init__(
        self,
        n_fft,
        n_hop,
        normalization_style="overall",
        input_is_spectrogram=False,
        hidden_size=512,
        phoneme_hidden_size=16, # hyperparameter
        nb_channels=2,
        sample_rate=16000,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        max_bin=None,
        unidirectional=False,
        power=1,
        number_of_phonemes = 65
    ):
        """
        Input:
        - mixture x (nb_samples, nb_channels, nb_timesteps)
                    or (nb_frames, nb_samples, nb_channels, nb_bins)
        - phoneme (nb_samples, nb_samples_phoneme_dim, nb_phonemes)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """
        
        super(OpenUnmix, self).__init__()
        
        self.nb_bins = n_fft // 2 + 1
        
        self.hidden_size = hidden_size
        
        self.stft = tf_transforms.STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = tf_transforms.Spectrogram(power=power)
        
        # model parameter, saved in state_dict but not trainable
        self.register_buffer('sample_rate', torch.tensor(sample_rate))
        self.sp_rate = sample_rate
        
        if input_is_spectrogram:
            self.transform = tf_transforms.NoOp()
        else:
            self.transform = nn.Sequential(self.stft, self.spec)
        
        self.normalize_input = normalization.Normalize(normalization_style,
                                                       input_mean,
                                                       input_scale,
                                                       self.nb_bins)
        
        self.fc1 = Linear(
            self.nb_bins*nb_channels, hidden_size,
            bias=False
        )
        
        self.bn1 = BatchNorm1d(hidden_size)
        
        if unidirectional:
            lstm_hidden_size = hidden_size+phoneme_hidden_size
        else:
            lstm_hidden_size = (hidden_size+phoneme_hidden_size) // 2
        
        self.lstm = LSTM(
            input_size=hidden_size+phoneme_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4
        )
        
        self.fc2 = Linear(
            in_features=(hidden_size+phoneme_hidden_size)*2,
            out_features=hidden_size,
            bias=False
        )
        
        self.bn2 = BatchNorm1d(hidden_size)
        
        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_bins*nb_channels,
            bias=False
        )
        
        self.bn3 = BatchNorm1d(self.nb_bins*nb_channels)
        
        self.output_scale = Parameter(
            torch.ones(self.nb_bins).float()
        )
        self.output_mean = Parameter(
            torch.ones(self.nb_bins).float()
        )
        

        self.phoneme_network = PhonemeNetwork(
                                    phoneme_hidden_size=phoneme_hidden_size,
                                    number_of_phonemes=number_of_phonemes,
                                    fft_window_duration=n_fft/self.sp_rate,
                                    fft_hop_duration=n_hop/self.sp_rate,
                                    center=False
                                )
    
    def forward(self, x, phoneme,offset=0):
        # check for waveform or spectrogram
        # transform to spectrogram if (nb_samples, nb_channels, nb_timesteps)
        # and reduce feature dimensions, therefore we reshape

        x = self.transform(x)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape
        
        mix = x.detach().clone()
        
        # shift and scale input to mean=0 std=1 (across all frames in one freq bin)
        x = self.normalize_input(x)
        
        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels*self.nb_bins))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range to [-1, 1]
        x = torch.tanh(x)
        
        """
        length = 1000
        begin = 4000
        plt.imshow(phoneme[0,begin:begin+length].detach().cpu().numpy(),
                    aspect=0.1,interpolation='none')
        plt.colorbar()
        plt.savefig('phoneme_input'+'.png',dpi=1200,bbox_inches='tight')
        plt.close("all")
        plt.clf()
        """
        
        # out [nb_frames,nb_samples,phoneme_hidden_size]
        phoneme = self.phoneme_network(phoneme,nb_frames,offset=offset)
        
        """
        plt.imshow(phoneme[begin:begin+length,0].detach().cpu().numpy(),
                    aspect=0.05,interpolation='none')
        plt.colorbar()
        plt.savefig('phoneme_distribution'+'.png',dpi=1200,bbox_inches='tight')
        plt.close("all")
        plt.clf()
        """
        
        # out [nb_frames, nb_samples,hidden_size+phoneme_hidden_size]
        x = torch.cat([x,phoneme],dim=-1)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)

        # first dense stage + batch norm
        #input to fc2 [nb_frames*nb_samples, 2*(hidden_size+phoneme_hidden_size)]
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean
        
        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix
        
        return x

if __name__ == '__main__':
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    umx = OpenUnmix(
        nb_channels=2,
        sample_rate=44100,
        max_bin = 1013
        ).to(device)
        
    #print(umx)
    data = torch.zeros((32,2,220500)).to(device)
    print(pytorch_model_summary.summary(umx, data, show_input=True))
    
    #summary(umx, input_data=data,batch_dim=1)