from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_model_summary
from torchsummary import summary

import normalization
import tf_transforms

class OpenUnmix(nn.Module):
    def __init__(
        self,
        normalization_style="overall",
        n_fft=4096,
        n_hop=1024,
        input_is_spectrogram=False,
        hidden_size=512,
        nb_channels=2,
        sample_rate=44100,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        max_bin=None,
        unidirectional=False,
        power=1,
        print=False
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        super(OpenUnmix, self).__init__()
        
        self.nb_output_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins
        

        self.hidden_size = hidden_size

        self.stft = tf_transforms.STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = tf_transforms.Spectrogram(power=power, mono=(nb_channels == 1))
        
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
                                                       self.nb_output_bins)
        
        self.fc1 = Linear(
            self.nb_bins*nb_channels, hidden_size,
            bias=False
        )

        self.bn1 = BatchNorm1d(hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4,
        )

        self.fc2 = Linear(
            in_features=hidden_size*2,
            out_features=hidden_size,
            bias=False
        )

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins*nb_channels,
            bias=False
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins*nb_channels)
        
        self.output_scale = Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        self.output_mean = Parameter(
            torch.ones(self.nb_output_bins).float()
        )

    def forward(self, x):
        # check for waveform or spectrogram
        # transform to spectrogram if (nb_samples, nb_channels, nb_timesteps)
        # and reduce feature dimensions, therefore we reshape
        x = self.transform(x)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()
        #print(mix.shape)
        
        # shift and scale input to mean=0 std=1 (across all frames in one freq bin)
        x = self.normalize_input(x)

        # crop, because we don't necessarily keep all bins due to the bandwidth
        x = x[..., :self.nb_bins]
        #print(x.shape)

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels*self.nb_bins))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)
        
        #print(x.shape)
        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean
        
        #print(x.shape)
        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix
        #print(x.shape)
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
    #demucs.forward(torch.Tensor(np.ones((1,2,220550))).to(device))
