#import sys
#sys.path.append("/home/felix/Documents/Mines/Césure/_Stage Télécom/open-unmix-pytorch/")
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_model_summary
from torchsummary import summary
import math

import normalization
import tf_transforms
from utils import checkValidConvolution

def conv_block(in_chans,out_chans):
    return nn.Sequential(
        nn.Conv2d(in_chans, out_chans, kernel_size=4,stride=2,padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(out_chans)
    )

def deconv_block(in_chans,out_chans,dropout=True,
                activation='relu',batchnorm=True):
    
    activations = nn.ModuleDict([
                ['sigmoid', nn.Sigmoid()],
                ['relu', nn.ReLU()]
    ])
    layers = [nn.ConvTranspose2d(in_chans,out_chans,kernel_size=4,stride=2,padding=1)]
            
    if dropout == True:
        layers.append(nn.Dropout2d(0.5))

    layers.append(activations[activation])
    
    if batchnorm == True:
        layers.append(nn.BatchNorm2d(out_chans))
    
    return nn.Sequential(*layers)

class Deep_u_net(nn.Module):
    def __init__(
        self,
        normalization_style="batch-specific",
        n_fft=4096,
        n_hop=1024,
        nb_channels=2,
        input_is_spectrogram=False,
        sample_rate=44100,
        print=False,
        input_mean=None,
        input_scale=None,
        max_bin=None
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        super(Deep_u_net, self).__init__()
        self.stft = tf_transforms.STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = tf_transforms.Spectrogram(power=1)
        self.print = print
        
        # register sample_rate to check at inference time
        self.register_buffer('sample_rate', torch.tensor(sample_rate))
        
        self.sp_rate = sample_rate

        if input_is_spectrogram:
            self.transform = tf_transforms.NoOp()
        else:
            self.transform = nn.Sequential(self.stft, self.spec)
        
        
        self.nb_output_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins
        
        self.normalize_input = normalization.Normalize(normalization_style,
                                                       input_mean,
                                                       input_scale,
                                                       self.nb_output_bins)
        self.encoder = nn.ModuleList()
        self.encoder.append(conv_block(nb_channels,16))
        
        for i in range(0,5):
            in_chans = 16*2**i
            self.encoder.append(conv_block(in_chans,2*in_chans))
        
        self.decoder = nn.ModuleList()
        self.decoder.append(deconv_block(16*2**5,16*2**4,dropout=True))
        for i in range(5,3,-1):
            in_chans = 16*2**i
            self.decoder.append(deconv_block(in_chans,in_chans//4,dropout=True))
        
        for i in range(3,1,-1):
            in_chans = 16*2**i
            self.decoder.append(deconv_block(in_chans,in_chans//4,dropout=False))
        
        self.decoder.append(deconv_block(16*2**1,nb_channels,dropout=False,activation='relu',batchnorm=False)) # stereo output
        
        

    def forward(self, mix):
        if self.print == True:print("original size:",mix.shape)
        
        # transform to spectrogram on the fly
        x = self.transform(mix)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape
        #print(nb_frames, nb_samples, nb_channels, nb_bins)
        
        if self.print == True:print("After transform:",x.shape)
        # reshape to the conventional shape for cnn in pytorch
        x = torch.reshape(x,(nb_samples,nb_channels,-1,nb_bins))
        
        x_original = x.detach().clone()
                
        x = self.normalize_input(x)
        
        x = x[...,:512] # keep 512 and not 513 frequency bins
        
        saved = []
        
        for encode in self.encoder:
            if self.print == True: print("Before encoding:",x.shape)
            saved.append(x)
            x = encode(x)
            #checkValidConvolution(x.shape[-1],kernel_size=4,stride=2,padding=1,note="encoder conv block")
            #checkValidConvolution(x.shape[-2],kernel_size=4,stride=2,padding=1,note="encoder conv block")
            if self.print == True: print("----CONVOL")
        
        if self.print == True: print("debut decoder:",x.shape)
        for decode in self.decoder:
            if self.print == True: print("Before decoding:",x.shape)

            x = decode(x)
            if self.print == True: print("After decoding:",x.shape)
            
            if len(saved) > 1:
                y = saved.pop(-1)
                x = torch.cat((x,y),1)
                if self.print == True: print("After skip connection:",x.shape)
            
            if self.print == True: print("----DECONVOL")
        
        # we go back to 513 bins and mutiply the original spectrogram by our mask
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ones = torch.ones(nb_samples,nb_channels,nb_frames,1).to(device)
        x = torch.cat((x,ones),3)
        x = x * x_original
        x = x.permute(2,0,1,3)
        #print("output:",x.shape)
        return x # return the magnitude spectrogram of the estimated source

if __name__ == '__main__':
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nb_channels = 1
    deep_u_net = Deep_u_net(
        nb_channels=nb_channels,
        sample_rate=8192,
        n_fft=1024,
        n_hop=768,
        print=True
        ).to(device)
    
    time = 98560
    mix = (torch.rand(16, nb_channels, time)+2)**2
    mix = mix.to(device)
    deep_u_net.forward(mix)
    
    #print(pytorch_model_summary.summary(deep_u_net, mix, show_input=False))
    
    #summary(deep_u_net, input_size=(2, 262144),batch_size=16)