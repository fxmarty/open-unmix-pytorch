#import sys
#sys.path.append("/home/felix/Documents/Mines/Césure/_Stage Télécom/open-unmix-pytorch/")
from model import Spectrogram, STFT, NoOp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_model_summary
from torchsummary import summary
import math
import normalization

def valid_length(length):
    """
    Return the nearest valid length to use with the model so that
    there is no time steps left over in a convolutions, e.g. for all
    layers, size of the (input - kernel_size) % stride = 0.

    For training, extracts should have a valid length.
    
    This is for testing purposes and is NOT used in the model.
    """
    for _ in range(6):
        length = math.ceil((length - 5) / 2) + 1
    for _ in range(6):
        length = (length - 1) * 2 + 5

    return int(length)

def conv_block(in_chans,out_chans):
    return nn.Sequential(
        nn.Conv2d(in_chans, out_chans, kernel_size=5,stride=2),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(out_chans)
    )

def deconv_block(in_chans,out_chans,dropout=True,
                activation='relu',batchnorm=True):
    
    activations = nn.ModuleDict([
                ['sigmoid', nn.Sigmoid()],
                ['relu', nn.ReLU()]
    ])
    layers = [nn.ConvTranspose2d(in_chans,out_chans,kernel_size=5,stride=2)]
            
    if dropout == True:
        layers.append(nn.Dropout2d(0.5))

    layers.append(activations[activation])
    
    if batchnorm == True:
        layers.append(nn.BatchNorm2d(out_chans))
    
    return nn.Sequential(*layers)
    
    # Division (input_size - kernel_size + padding)/stride must be an integer
def padPerfectly(kernel_size,heigh,width,stride):
    padding_h = 0
    padding_w = 0
    
    if (heigh - kernel_size) % stride != 0:
        # Compute the division above without padding (which may be a float)
        divisionValue_idealPadding_h = (heigh - kernel_size)/stride
        padding_h = math.ceil(
                    divisionValue_idealPadding_h*stride/(heigh - kernel_size))
        
    if (width - kernel_size) % stride != 0:
        divisionValue_idealPadding_w = (width - kernel_size)/stride
        padding_w = math.ceil(
                    divisionValue_idealPadding_w*stride/(width - kernel_size))
    
    # Reverse heigh and width order according to F.pad behavior
    return (padding_w//2,(padding_w - padding_w//2),
            padding_h//2,(padding_h - padding_h//2))
    
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
        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=1, mono=(nb_channels == 1))
        self.print = print
        
        # register sample_rate to check at inference time
        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        if input_is_spectrogram:
            self.transform = NoOp()
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
                                                       self.nb_bins)
        
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
        
        self.decoder.append(deconv_block(16*2**1,nb_channels,dropout=False,activation="sigmoid",batchnorm=False)) # stereo output
    

    def forward(self, mix):
        # transform to spectrogram on the fly
        x = self.transform(mix)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape
        
        # reshape to the conventional shape for cnn in pytorch
        x = torch.reshape(x,(nb_samples,nb_channels,-1,nb_bins))
        
        x_original = x.detach().clone()
        
        # scale between 0 and 1
        #xmax = torch.max(x)
        #xmin = torch.min(x)
        #x = (x - xmin)/(xmax-xmin)
        
        x = self.normalize_input(x)
        
        saved = []
        saved_pad = []
        
        for encode in self.encoder:
            perfectPad = padPerfectly(5,x.shape[-2],x.shape[-1],2)
            saved_pad.append(perfectPad)
            
            if self.print == True: print("Before padding encoder:",x.shape)
            x = F.pad(x,perfectPad)
            saved.append(x)
            if self.print == True: print("After padding encoder:",x.shape)
            
            x = encode(x)
            if self.print == True: print("----CONVOL")
        
        saved_pad.append((0,0,0,0)) # after the last convolution, we do not pad

        if self.print == True: print("debut decoder:",x.shape)
        for decode in self.decoder:
            pad_w_l,pad_w_r,pad_h_l,pad_h_r = saved_pad.pop()
            
            if self.print == True: print("Before slicing:",x.shape)
            x = x[...,pad_h_l:-pad_h_r or None,pad_w_l:-pad_w_r or None]
            if self.print == True: print("After slicing:",x.shape)

            x = decode(x)
            if self.print == True: print("After deconv:",x.shape)
            
            if len(saved) > 1:
                y = saved.pop(-1)
                x = torch.cat((x,y),1)
                if self.print == True: print("After skip connection:",x.shape)
            
            if self.print == True: print("----DECONVOL")
        
        pad_w_l,pad_w_r,pad_h_l,pad_h_r = saved_pad.pop()
        if self.print == True: print("Before slicing:",x.shape)
        x = x[...,pad_h_l:-pad_h_r or None,pad_w_l:-pad_w_r or None]
        if self.print == True: print("After slicing:",x.shape)

        x = x * x_original
        x = x.permute(2,0,1,3)
        return x # return the magnitude spectrogram of the estimated source

if __name__ == '__main__':
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deep_u_net = Deep_u_net(
        nb_channels=2,
        sample_rate=44100,
        print=True
        ).to(device)
        
    #print(deep_u_net)    
    mix = (torch.rand(1, 2, 262144)+2)**2
    mix = mix.to(device)
    deep_u_net.forward(mix)
    
    #print(pytorch_model_summary.summary(deep_u_net, mix, show_input=False))
    
    #summary(deep_u_net, input_size=(2, 262144),batch_size=16)