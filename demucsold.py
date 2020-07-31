from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary

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
        
        # register sample_rate to check at inference time
        self.register_buffer('sample_rate', torch.tensor(sample_rate))
        
        self.conv1 = nn.Conv1d(nb_channels, 64, kernel_size=8,stride=4)
        self.conv1pixel = nn.Conv1d(64, 128, kernel_size=1,stride=1)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8,stride=4)
        self.conv2pixel = nn.Conv1d(128, 256, kernel_size=1,stride=1)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=8,stride=4)
        self.conv3pixel = nn.Conv1d(256, 512, kernel_size=1,stride=1)
        
        self.conv4 = nn.Conv1d(256, 512, kernel_size=8,stride=4)
        self.conv4pixel = nn.Conv1d(512, 1024, kernel_size=1,stride=1)
        
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=8,stride=4)
        self.conv5pixel = nn.Conv1d(1024, 2048, kernel_size=1,stride=1)
        
        self.conv6 = nn.Conv1d(1024, 2048, kernel_size=8,stride=4)
        self.conv6pixel = nn.Conv1d(2048, 4096, kernel_size=1,stride=1)
        
        """
        self.lstm = LSTM(
            input_size=hidden_size, #???
            hidden_size=1536,
            num_layers=2,
            bidirectional=not unidirectional,
            batch_first=False
        )
        """
        
    def forward(self, mix):
        # transform to spectrogram on the fly
        #nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape
        print(mix.shape)
        
        x = F.relu(self.conv1(mix))
        print(x.shape)
        
        encoder1 = F.glu(self.conv1pixel(x),dim=-2) # axis?
        print(encoder1.shape)
        
        x = F.relu(self.conv2(encoder1))
        encoder2 = F.glu(self.conv2pixel(x),dim=-2) # axis?
        print(encoder2.shape)
        
        
        x = F.relu(self.conv3(encoder2))
        encoder3 = F.glu(self.conv3pixel(x),dim=-2) # axis?
        print(encoder3.shape)
        
        
        x = F.relu(self.conv4(encoder3))
        encoder4 = F.glu(self.conv4pixel(x),dim=-2) # axis?
        print(encoder4.shape)
        
        
        x = F.relu(self.conv5(encoder4))
        encoder5 = F.glu(self.conv5pixel(x),dim=-2) # axis?
        print(encoder5.shape)
        
        
        x = F.relu(self.conv6(encoder5))
        encoder6 = F.glu(self.conv6pixel(x),dim=-2) # axis?
        
        #x = self.lstm(encoder6)

        return encoder6

if __name__ == '__main__':
    from torch.autograd import Variable
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demucs = Demucs(
        nb_channels=2,
        sample_rate=44100
        ).to(device)
        
    print(demucs)
    print(summary(demucs, torch.zeros((1,2,220500)).to(device), show_input=False))
    
    #unet.forward(Variable(torch.Tensor(np.ones((7,1,512,128)))))