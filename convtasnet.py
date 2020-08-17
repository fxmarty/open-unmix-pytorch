import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_model_summary
from torchsummary import summary
from utils import checkValidConvolution, valid_length, memory_check

import normalization

class TemporalBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 sc_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation):
        super(TemporalBlock, self).__init__()

        
        self.kernel_size = kernel_size
        self.stride,self.padding,self.dilation = stride,padding,dilation
        
        self.conv1x1 = nn.Conv1d(in_channels, hidden_channels, 1, bias=False)
        
        self.nonlinearity1 = nn.PReLU()
        
        self.norm1 = nn.GroupNorm(1, hidden_channels, eps=1e-08)
        
        
        self.depthwise_conv = nn.Conv1d(hidden_channels,
                                   hidden_channels,
                                   kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_channels,
                                   bias=False)
        
        self.nonlinearity2 = nn.PReLU()
        
        self.norm2 = nn.GroupNorm(1, hidden_channels, eps=1e-08)
        
        self.skip_out = nn.Conv1d(hidden_channels, sc_channels, kernel_size=1)
        self.res_out = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)


    def forward(self, input):
        #checkValidConvolution(input.size(2),1)
        #memory_check("inside TemporalBlock")
        output = self.norm1(self.nonlinearity1(self.conv1x1(input)))
        #memory_check("inside TemporalBlock")
        #checkValidConvolution(output.size(2),self.kernel_size,self.stride,self.padding,self.dilation)
        output = self.norm2(self.nonlinearity2(self.depthwise_conv(output)))
        #memory_check("inside TemporalBlock")
        
        #checkValidConvolution(output.size(2),self.kernel_size)
        residual = self.res_out(output)
        skip = self.skip_out(output)
        #memory_check("inside TemporalBlock")
        return residual, skip


class ConvTasNet(nn.Module):
    def __init__(
        self,
        normalization_style='none',
        nb_channels=2,
        sample_rate=44100,
        print=False,
        N=512,
        L=16,
        B=128,
        H=512,
        Sc=128, # to modify
        P=3,
        X=8,
        R=3,
        C=4
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output: (nb_samples, nb_channels, nb_timesteps)
        
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
        self.print = print
        
        self.normalize_input = normalization.Normalize(normalization_style)
        
        # Encoder part
        """
        self.encoder = nn.Sequential(
                                nn.Conv1d(nb_channels,N,kernel_size=L,stride=L // 2),
                                nn.ReLU()
                                )
        """
        
        self.encoder = nn.Conv1d(nb_channels,N,kernel_size=L,stride=L // 2)
        
        # Separation part
        self.layerNorm = nn.GroupNorm(1, N, eps=1e-8)
        # [M, N, K] -> [M, B, K]
        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        
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
                                            dilation=dilation)
                            )
            self.repeats.append(repeat)
        
        # [M, B, K] -> [M, C*N, K]
        
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(Sc, C * N, 1),
                                    nn.Sigmoid()
                                   )
        # Decoder part
        self.decoder = nn.ConvTranspose1d(N, nb_channels, kernel_size=L,stride=L // 2, bias=False)
            
    def pad_signal(self,input):
        use_cuda = torch.cuda.is_available() #overrides no-cuda parameter. Take care!
        device = torch.device("cuda" if use_cuda else "cpu")
        
        # input is the waveforms: (batch_size,nb_channels,T)
        batch_size = input.size(0)
        nb_channels = input.size(1)
        nsample = input.size(2)
        
        # pad at least 8 each side
        ideal_size = valid_length(nsample+2*self.L//2,self.L,stride=self.L//2)
        padding_size = ideal_size - nsample 
        
        pad_aux_left = torch.zeros(batch_size,nb_channels,padding_size//2).to(device)
        pad_aux_right = torch.zeros(batch_size,nb_channels,padding_size - padding_size//2).to(device)
        input = torch.cat([pad_aux_left, input, pad_aux_right], 2)
        
        return input, padding_size

    def forward(self, mix):
        
        memory_check("A debut de forward:")
        
        if self.print == True: print(mix.shape)
        # Padding
        mix,padding_size = self.pad_signal(mix)
        memory_check("After padding:")
        # Input normalization
        mix = self.normalize_input(mix) # does nothing by default
        memory_check("After normalization:")
        # Encoder
        if self.print == True: print(mix.shape)
        #checkValidConvolution(mix.size(2),kernel_size=self.L,stride=self.L // 2,note="encoder")
        mix_encoded = self.encoder(mix)
        if self.print == True: print(mix_encoded.shape)
        
        memory_check("after encoder:")
        x = self.layerNorm(mix_encoded)
        if self.print == True: print(x.shape)
        
        #checkValidConvolution(x.size(2),kernel_size=1)
        output = self.bottleneck_conv1x1(x)
        del x
        skip_connection = 0.
        if self.print == True: print(output.shape)
        
        memory_check("before repeat blocks:")
        i = 0
        for repeat in self.repeats:
            for temporalBlock in repeat:
                memory_check(str(i) + "SAME:")
                residual, skip = temporalBlock(output)
                #memory_check(str(i) + "after temporalBlock:")
                skip_connection = skip_connection + skip
                #memory_check(str(i) + ":")
                if self.print == True: print("skip_connection size:",skip_connection.shape)
                output = output + residual
                #memory_check(str(i) + ":")
                if self.print == True: print(i,":",output.shape)
                i=i+1
            
        
        #checkValidConvolution(skip_connection.size(2),kernel_size=1)
        x = self.output(skip_connection)
        if self.print == True: print(x.shape)
        masks = x.view(-1, self.C, self.N, x.shape[-1])
        if self.print == True: print("masks:",masks.shape)
        
        memory_check("Presque avant fin :")
        mix_encoded = torch.unsqueeze(mix_encoded,1)
        if self.print == True: print("mix_encoded:",mix_encoded.shape)
        
        x = masks * mix_encoded
        if self.print == True: print("ok",x.shape)
        
        x = x.view(x.shape[0]*self.C,self.N,-1)
        if self.print == True: print(x.shape)
        x = self.decoder(x)
        if self.print == True: print(x.shape)
        x = x[:,:,padding_size//2:-(padding_size - padding_size//2)]#.contiguous()  
        if self.print == True: print(x.shape)
        memory_check("A la fin :")
        return x

if __name__ == '__main__':
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvTasNet(print=False,nb_channels=1).to(device)
    
    taille = int(1.5*44100)
    #print(deep_u_net)    
    mix = (torch.rand(4, 1, taille)+2)**2
    mix = mix.to(device)
    model.forward(mix)
    
    #model.pad_signal(mix)
    
    
    #print(pytorch_model_summary.summary(model, mix, show_input=False))
    
    #summary(model, input_data=(2, 262144))