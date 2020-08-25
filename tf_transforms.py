import torch
import torch.nn as nn

class NoOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class STFT(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        center=False
    ):
        super(STFT, self).__init__()
        self.window = nn.Parameter(
            torch.hann_window(n_fft),
            requires_grad=False
        )
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """
        #print(x.shape)
        
        nb_samples, nb_channels, nb_timesteps = x.size()
        
        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_samples*nb_channels, -1)
        #print(x.shape)
        # compute stft with parameters as close as possible scipy settings
        
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft, hop_length=self.n_hop,
            window=self.window, center=self.center,
            normalized=False, onesided=True,
            pad_mode='reflect'
        )
        # Dim (nb_channels,nb_bin_freq,nb_frames,2)
        
        # reshape back to channel dimension
        stft_f = stft_f.contiguous().view(
            nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2
        )

        return stft_f


class Spectrogram(nn.Module):
    def __init__(
        self,
        power=1,
        mono=True
    ):
        super(Spectrogram, self).__init__()
        self.power = power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram
            (nb_frames, nb_samples, nb_channels, nb_bins)
        """
        stft_f = stft_f.transpose(2, 3) # put nb_frames before nb_bins
        # take the magnitude
        # -1 for the last column, we don't take the spectrogram at a power of 2
        stft_f = stft_f.pow(2).sum(-1).pow(self.power / 2.0)
        
        
        # If nb_channels = 1, downmix in the mag domain, A MODIFIER
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)
        
        # permute output for LSTM convenience
        return stft_f.permute(2, 0, 1, 3)