import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary

def conv_block(in_chans,out_chans):
    return nn.Sequential(
        nn.Conv1d(in_chans, out_chans,kernel_size=8,stride=4),
        nn.ReLU(),
        nn.Conv1d(out_chans,2*out_chans,kernel_size=1,stride=1),
        nn.GLU(dim=-2)
    )

def trim(A, reference): #trim A on last axis to match reference shape
    difference = A.shape[-1] - reference.shape[-1]
    if difference < 0:
        raise ValueError("Reference should be smaller than the tensor to be trimmed!")
    if difference > 0:
            return A[...,difference//2:-(difference - difference // 2)]

"""
class BLSTM(nn.Module):
    def __init__(self, dim, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        return x
"""

class Demucs(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        nb_channels=2,
        input_is_spectrogram=False,
        sample_rate=44100,
    ):
        """
        Input:  (batch, channel, sample)
            or  (frame, batch, channels, frequency)
        Output: (frame, batch, channels, frequency)
        """
        super(Demucs, self).__init__()
        
        self.encoder = nn.ModuleList()
        
        # register sample_rate to check at inference time
        self.register_buffer('sample_rate', torch.tensor(sample_rate))
        
        self.encoder.append(conv_block(2,64)) # 2 for stereo
        for i in range(0,5):
            in_chans = 64*2**i
            self.encoder.append(conv_block(in_chans,2*in_chans))
        
        self.lstm = nn.LSTM(bidirectional=True, num_layers=2, hidden_size=64*2**5, input_size=64*2**5,batch_first=False)
        
        self.pixel = nn.Conv1d(2*(64*2**5),64*2**5,kernel_size=1,stride=1)
        
        self.decoder = nn.ModuleList()
        for i in range(5,-1,-1):
            in_chans = 64*2**i
            out_chans = in_chans // 2
            
            if i == 0:
                out_chans = 2 # stereo output
            
            deconv_block_before = nn.Sequential(
                nn.Conv1d(in_chans,in_chans,kernel_size=3,stride=1),
                nn.ReLU()
            )
            deconv_block_after = nn.Sequential(
                nn.Conv1d(2*in_chans,2*in_chans,kernel_size=1,stride=1),
                nn.GLU(dim=-2),
                nn.ConvTranspose1d(in_chans,out_chans,kernel_size=8,stride=4),
                nn.ReLU()
            )
            
            self.decoder.append(nn.ModuleList([deconv_block_before,deconv_block_after]))
            

    def forward(self, mix):
        x = mix
        print(x.shape)
        saved = [x]
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)
            print(x.shape)
        
        x = x.permute(2, 0, 1) # to fit LSTM expected shape
        x = self.lstm(x)[0]
        x = x.permute(1, 2, 0)
        x = self.pixel(x)
        
        print(x.shape)
        for decodeList in self.decoder:
            deconvBlockBeforeConcat = decodeList[0]
            deconvBlockAfterConcat = decodeList[1]
            x = deconvBlockBeforeConcat(x)
            
            convPathResult = trim(saved.pop(-1),x)
            x = torch.cat((x,convPathResult),1)
            x = deconvBlockAfterConcat(x)
            print(x.shape)
            
        return x

if __name__ == '__main__':
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demucs = Demucs(
        nb_channels=2,
        sample_rate=44100
        ).to(device)
        
    print(demucs)
    #print(summary(demucs, torch.zeros((1,2,220500)).to(device), show_input=False))
    
    demucs.forward(torch.Tensor(np.ones((1,2,220550))).to(device))