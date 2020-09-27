import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_model_summary
from torchsummary import summary

import torchsnooper

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from utils import checkValidConvolution, valid_length, memory_check
import time_transform_posteriograms
import normalization

class PhonemeNetwork(nn.Module):
    def __init__(self,
                L,
                number_of_phonemes,
                phoneme_hidden_size,
                N,
                sp_rate,
                single_phoneme, # indicates of the phoneme map is of 64 or 1
                filter_length=5 # has to be odd
                ):
        
        super(PhonemeNetwork, self).__init__()
        
        self.sp_rate = sp_rate
        self.encoder_kernel = L/sp_rate
        self.encoder_stride = self.encoder_kernel/2
        self.single_phoneme = single_phoneme
        
        self.embedding = nn.Sequential(
                            nn.Embedding(number_of_phonemes+1,
                                         phoneme_hidden_size,
                                         padding_idx=0),
                            nn.ReLU(),
                            nn.Dropout(p=0.5)
                        )
        
        self.convol = nn.Sequential(
                        nn.Conv1d(in_channels=phoneme_hidden_size,
                              out_channels=phoneme_hidden_size,
                              kernel_size=filter_length,
                              stride=1,
                              padding=filter_length//2
                                ),
                        nn.ReLU()
                        )
        
        self.fc_weights = nn.Sequential(
                            nn.Linear(phoneme_hidden_size, N),
                            nn.ReLU()
                          )
                          
        self.fc_bias = nn.Sequential(
                            nn.Linear(phoneme_hidden_size, N),
                            nn.ReLU()
                       )
                
        """
        Input: phoneme posteriogram (nb_time_frames_phoneme, nb_phonemes)

        """
    def forward(self, phoneme,nb_frames,padding_size):
        
        # out [nb_samples, nb_frames]
        if self.single_phoneme:
            phoneme = time_transform_posteriograms.conv_tasnet_single(
                                        phoneme,nb_frames,
                                        padding_size,0.016,self.encoder_kernel,
                                        self.encoder_stride,sp_rate=self.sp_rate)
        
        else:
            raise ValueError("Multiple phoneme version has to be implemented for CTN")
        
        # out [nb_samples, nb_frames, phoneme_hidden_size]
        phoneme = self.embedding(phoneme)

        # to adapt to convolution need, reshape to
        # [nb_samples, phoneme_hidden_size, nb_frames]
        phoneme = torch.transpose(phoneme,1,2)

        # out [nb_samples, phoneme_hidden_size, nb_frames]
        phoneme = self.convol(phoneme)

        # out [nb_samples, nb_frames, phoneme_hidden_size]
        phoneme = torch.transpose(phoneme,1,2)

        # out 2 * [nb_samples, nb_frames, N]
        weights = self.fc_weights(phoneme)
        bias = self.fc_weights(phoneme)

        # transpose for consistency with the main separation model
        weights = torch.transpose(weights,-1,-2)
        bias = torch.transpose(bias,-1,-2)
        
        # out 2 * [nb_samples, N, nb_frames]
        return weights,bias

# The temporal block preserves the length of the signal received as input
class TemporalBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 sc_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 end_block): # end block for having only skip at the end
        super(TemporalBlock, self).__init__()

        self.kernel_size = kernel_size
        self.end_block = end_block
        
        self.stride,self.padding,self.dilation = stride,padding,dilation
        
        self.conv1x1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        
        self.nonlinearity1 = nn.PReLU()
        
        self.norm1 = nn.GroupNorm(1, hidden_channels, eps=1e-08)
        
        self.depthwise_conv = nn.Conv1d(hidden_channels,
                                   hidden_channels,
                                   kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=hidden_channels
                                   )
        
        self.nonlinearity2 = nn.PReLU()
        
        self.norm2 = nn.GroupNorm(1, hidden_channels, eps=1e-08)
        
        self.skip_out = nn.Conv1d(hidden_channels, sc_channels, kernel_size=1)
        
        if end_block == False:
            self.res_out = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, input):
        #checkValidConvolution(input.size(2),1)
        output = self.norm1(self.nonlinearity1(self.conv1x1(input)))
        #checkValidConvolution(output.size(2),self.kernel_size,self.stride,self.padding,self.dilation)
        output = self.norm2(self.nonlinearity2(self.depthwise_conv(output)))
        
        #checkValidConvolution(output.size(2),self.kernel_size)
        
        skip = self.skip_out(output)
        
        if self.end_block == False:
            residual = self.res_out(output)
            return residual, skip
        else:
            return skip


class ConvTasNet(nn.Module):
    def __init__(
        self,
        normalization_style='none',
        nb_channels=2,
        sample_rate=44100,
        N=128, # Originally 512
        L=16, # Originally 16, but 20 for 44100 Hz
        B=128,
        H=256, # Originally 512
        Sc=128,
        P=3,
        X=8, # Originally 8 ; 10 for same receptive field at 44100 Hz, 9 for 16000 Hz
        R=2, # Originally 3
        C=1, # We usually only extract one source
        number_of_phonemes=64,
        phoneme_hidden_size=50,
        single_phoneme=False
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output: (nb_samples, C, nb_channels, nb_timesteps)
        
        Parameters
        ----------
        N : int
            Number of channels after the encoder
        L :
            Length of the filters (in samples, e.g. L/fs = sequence length in second)
        B : int
            Number of channels in the separation block, after bottleneck layer
        H : int
            Number of channels in a 1-D Conv block
        Sc : int
            Number of channel at the output of the skip connection path
        P : int
            Kernel size in convolutional blocks
        X : int
            Number of convolutional blocks in each repeat
        R : int
            Number of repeat
        C : int
            Number of speakers
        """
        super(ConvTasNet, self).__init__()
        
        self.C = C
        self.N = N
        self.L = L
        self.X = X
        self.R = R
        self.register_buffer('sample_rate', torch.tensor(sample_rate))
        self.sp_rate = sample_rate
        self.nb_channels = nb_channels
        self.number_of_phonemes = number_of_phonemes
        
        self.normalize_input = normalization.Normalize(normalization_style)
        
        # Encoder part
        self.encoder = nn.Sequential(
                            nn.Conv1d(nb_channels,N,kernel_size=L,stride=L // 2,bias=False),
                            nn.ReLU()
                        )
        
        # Separation part
        self.layerNorm = nn.GroupNorm(1, N, eps=1e-8)
        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1)
        
        self.phoneme_network = PhonemeNetwork(L=L,
                                              number_of_phonemes=number_of_phonemes,
                                              phoneme_hidden_size=phoneme_hidden_size,
                                              N=N,
                                              sp_rate=self.sp_rate,
                                              single_phoneme=single_phoneme)
        
        self.repeats = nn.ModuleList()
        
        for r in range(R):
            blocks = []
            repeat = nn.ModuleList()
            for x in range(X):
                dilation = 2**x
                padding = (P - 1) * dilation // 2
                repeat.append(TemporalBlock(in_channels=B,
                                            hidden_channels=H,
                                            sc_channels=Sc,
                                            kernel_size=P,
                                            stride=1,
                                            padding=padding,
                                            dilation=dilation,
                                            end_block=(r == R - 1 and x == X - 1))
                            )
            self.repeats.append(repeat)
                
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(Sc, C * N, kernel_size=1),
                                    nn.Sigmoid()
                                   )
        # Decoder part
        self.decoder = nn.ConvTranspose1d(N, nb_channels, kernel_size=L,stride=L // 2, bias=False)
            
    def pad_signal(self,input):        
        # input is the waveforms: (batch_size,nb_channels,T)
        batch_size = input.size(0)
        nb_channels = input.size(1)
        nsample = input.size(2)
        
        # pad at least self.L//2 each side
        ideal_size = valid_length(nsample+2*self.L//2,self.L,stride=self.L//2)
        padding_size = ideal_size - nsample 
        
        pad_aux_left = torch.zeros(batch_size,nb_channels,padding_size//2).to(input.device)
        pad_aux_right = torch.zeros(batch_size,nb_channels,padding_size - padding_size//2).to(input.device)
        output = torch.cat([pad_aux_left, input, pad_aux_right], 2)
        return output, padding_size
    
    
    def forward(self, x, phoneme):
        nb_samples = x.shape[0]
        
        # Padding, output [nb_samples, nb_channels, nb_timesteps_ideal]
        x,padding_size = self.pad_signal(x)
        
        # Input normalization
        x = self.normalize_input(x) # does nothing by default
        
        # Encoder
        #checkValidConvolution(x.size(2),kernel_size=self.L,stride=self.L // 2,note="encoder")
        x = self.encoder(x)
        nb_samples, N, hidden_size = x.size()
        
        mix_encoded = x.clone()
        
        # out 2 * [nb_samples, N, nb_frames]
        phoneme_weights, phoneme_bias = self.phoneme_network(
                                        phoneme,hidden_size,padding_size)
        
        x = phoneme_weights * x + phoneme_bias
        
        # Separation
        x = self.layerNorm(x) # output [nb_samples, N, hidden_size]
        
        #checkValidConvolution(x.size(2),kernel_size=1)
        x = self.bottleneck_conv1x1(x) #output [nb_samples, B, hidden_size]
        
        skip_connection = 0.
        i = 1
        for repeat in self.repeats:
            for temporalBlock in repeat:
                # residual [nb_samples, Sc, hidden_size]
                # skip [nb_samples, B, hidden_size]
                if i < self.R*self.X:
                    residual, skip = temporalBlock(x)
                    skip_connection = skip_connection + skip
                    x = x + residual
                else:
                    skip = temporalBlock(x)
                    skip_connection = skip_connection + skip
                i=i+1
            
        x = self.output(skip_connection) # output [nb_samples, C*N, hidden_size]
        
        # output [nb_samples, C, N, hidden_size]
        masks = x.view(-1, self.C, self.N, x.shape[-1]) 
        
        mix_encoded = torch.unsqueeze(mix_encoded,1)
        
        x = masks * mix_encoded # output [nb_samples, C, N, hidden_size]
        
        x = x.view(x.shape[0]*self.C,self.N,-1) # output [nb_samples * C, N, hidden_size]
        x = self.decoder(x) # output [nb_samples * C, nb_channels, nb_timesteps_ideal]
        x = x.view(nb_samples,self.C,self.nb_channels,-1)
        # output [nb_samples, C, nb_channels, nb_timesteps_ideal]
        
        # output padding [nb_samples, C, nb_channels, nb_timesteps]
        x = x[:,:,:,padding_size//2:-(padding_size - padding_size//2)]  
        return x

if __name__ == '__main__':
    import numpy as np
    from torchviz import make_dot
    import hiddenlayer as hl
    #import IPython
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvTasNet(nb_channels=1,C=2).to(device)
    model.eval()
    
    taille = int(2.2*44100)
    #print(deep_u_net)    
    mix = (torch.rand(4, 1, taille)+2)#.detach()
    mix = mix.to(device)
    #with torch.no_grad():
    res = model(mix)   
    #make_dot(res,params=model.named_parameters()).render("attached", format="png")
    make_dot(res,params={**{'inputs': mix}, **dict(model.named_parameters())}).render("attached", format="png")
    """
    im = hl.build_graph(model, mix)
    dot=im.build_dot()
    dot.format="png"
    im=dot.render("attached_hl",cleanup=True)
    """
    
    #print(pytorch_model_summary.summary(model, mix, show_input=False))
    
    #summary(model, input_data=(2, 262144))