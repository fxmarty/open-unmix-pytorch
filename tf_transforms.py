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
        power=1
    ):
        super(Spectrogram, self).__init__()
        self.power = power

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
        
        # permute output for LSTM convenience
        return stft_f.permute(2, 0, 1, 3)

if __name__ == '__main__':
    
    import utils
    import test
    
    data = torch.zeros((1,2,220500))
    print(data.shape)
    
    pad_length = 0
    data,pad_length = utils.pad_for_stft(data, 1024)
    print(data.shape)
    
    stftNoCenter = STFT(center=False)
    stftWithCenter = STFT(center=True)
        
    res_noCenter = stftNoCenter(data).cpu().numpy()
    res_withCenter = stftWithCenter(data).cpu().numpy()
    
    res_noCenter = res_noCenter[..., 0] + res_noCenter[..., 1]*1j
    res_noCenter = res_noCenter[0].transpose(2, 1, 0)
    
    res_withCenter = res_withCenter[..., 0] + res_withCenter[..., 1]*1j
    res_withCenter = res_withCenter[0].transpose(2, 1, 0)
    
    padding = int(4096//2)
    
    th_nopad = (data.shape[-1] - (4096 - 1) - 1)/1024 + 1
    
    th_pad = (data.shape[-1] + 2*padding - (4096 - 1) - 1)/1024 + 1
    print("Theorical output without Center:",th_nopad)
    print("Theorical output shape with Center:",th_pad)
    
    
    print("No center",res_noCenter.shape)
    print("With center",res_withCenter.shape)
    
    print("-- istft --")
    istft_noCenter = test.istft(res_noCenter.T)
    istft_withCenter = test.istft(res_withCenter.T)
    
    print("No center",istft_noCenter.shape)
    print("With center",istft_withCenter.shape)
    
    if pad_length > 0:
        print("-- istft unpadded --")
        print("No center",istft_noCenter[...,:-pad_length].shape)
        print("With center",istft_withCenter[...,:-pad_length].shape)
    
    
    