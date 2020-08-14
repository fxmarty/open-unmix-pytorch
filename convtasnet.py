import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_model_summary
from torchsummary import summary

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
        # [M, B, K] -> [M, H, K]
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
        output = self.norm1(self.nonlinearity1(self.conv1x1(input)))

        output = self.norm2(self.nonlinearity2(self.depthwise_conv(output)))
        
        residual = self.res_out(output)

        skip = self.skip_out(output)
        return residual, skip


class ConvTasNet(nn.Module):
    def __init__(
        self,
        nb_channels=2,
        sample_rate=44100,
        print=False,
        N=512,
        L=16,
        B=128,
        H=512,
        Sc=256, # to modify
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
        self.C = C
        self.N = N
        super(ConvTasNet, self).__init__()
        
        # Encoder part
        self.encoder = nn.Sequential(
                                nn.Conv1d(nb_channels,N,kernel_size=L,stride=L // 2),
                                nn.ReLU()
                                )
        
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
            

    def forward(self, mix):
        
        print(mix.shape)
        mix_encoded = self.encoder(mix)
        print(mix_encoded.shape)
        
        x = self.layerNorm(mix_encoded)
        print(x.shape)
        
        output = self.bottleneck_conv1x1(x)
        skip_connection = 0.
        print(output.shape)
        
        i = 0
        for repeat in self.repeats:
            for temporalBlock in repeat:
                residual, skip = temporalBlock(output)
                skip_connection = skip_connection + skip
                output = output + residual
                print(i,":",output.shape)
                i=i+1
            
        x = self.output(skip_connection)
        print(x.shape)
        masks = x.view(-1, self.C, self.N, x.shape[-1])
        print("masks:",masks.shape)
        
        mix_encoded = torch.unsqueeze(mix_encoded,1)
        print("mix_encoded:",mix_encoded.shape)
        
        x = masks * mix_encoded
        print("ok",x.shape)
        
        x = x.view(x.shape[0]*self.C,self.N,-1)
        print(x.shape)
        x = self.decoder(x)
        print(x.shape)
        return x

if __name__ == '__main__':
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvTasNet(
        ).to(device)
        
    #print(deep_u_net)    
    mix = (torch.rand(1, 2, 15855)+2)**2
    mix = mix.to(device)
    model.forward(mix)
    
    #print(pytorch_model_summary.summary(deep_u_net, mix, show_input=False))
    
    #summary(deep_u_net, input_size=(2, 262144),batch_size=16)