#import sys
#sys.path.append("/home/felix/Documents/Mines/Césure/_Stage Télécom/open-unmix-pytorch/")
from model import Spectrogram, STFT, NoOp
from torch.nn import LSTM, Linear, BatchNorm2d, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F

class deep_u_net(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        nb_channels=2,
        max_bin=None,
        input_is_spectrogram=False,
        sample_rate=44100,
    ):
        """
        Input:  (batch, channel, sample)
            or  (frame, batch, channels, frequency)
        Output: (frame, batch, channels, frequency)
        """

        super(deep_u_net, self).__init__()
        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=1, mono=(nb_channels == 1))
        # register sample_rate to check at inference time
        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        if input_is_spectrogram:
            self.transform = NoOp()
        else:
            self.transform = nn.Sequential(self.stft, self.spec)
        
        self.nb_output_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin # Bandwidth should be 8192 Hz according to the
            # paper, very arbitrary choice.
        else:
            self.nb_bins = self.nb_output_bins
        
        self.conv1 = nn.Conv2d(nb_channels, 16, kernel_size=5,stride=2,padding=2)
        self.bn1 = BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5,stride=2,padding=2)
        self.bn2 = BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5,stride=2,padding=2)
        self.bn3 = BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5,stride=2,padding=2)
        self.bn4 = BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=5,stride=2,padding=2)
        self.bn5 = BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(256, 512, kernel_size=5,stride=2,padding=2)
        self.bn6 = BatchNorm2d(512)
        
        self.deconv7 = nn.ConvTranspose2d(512,256,kernel_size=5,stride=2,padding=2,output_padding=1)
        self.bn7 = BatchNorm2d(256)
        
        self.dropout1 = nn.Dropout2d(0.5)
        
        self.deconv8 = nn.ConvTranspose2d(512,128,kernel_size=5,stride=2,padding=2,output_padding=1)
        self.bn8 = BatchNorm2d(128)
        
        self.deconv9 = nn.ConvTranspose2d(256,64,kernel_size=5,stride=2,padding=2,output_padding=1)
        self.bn9 = BatchNorm2d(64)
        
        self.deconv10 = nn.ConvTranspose2d(128,32,kernel_size=5,stride=2,padding=2,output_padding=1)
        self.bn10 = BatchNorm2d(32)
        
        self.deconv11 = nn.ConvTranspose2d(64,16,kernel_size=5,stride=2,padding=2,output_padding=1)
        self.bn11 = BatchNorm2d(16)
        
        self.deconv12 = nn.ConvTranspose2d(32,1,kernel_size=5,stride=2,padding=2,output_padding=1)
        self.bn12 = BatchNorm2d(1)

    def forward(self, mix):
        
        # transform to spectrogram on the fly
        x = self.transform(mix)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape
        
        # Size according to the paper, just for the testing (to be deleted)
        x = x[:,:,:128,:512]
        
        x_original = x.detach().clone()
        
        # scale between 0 and 1
        xmax = torch.max(x)
        xmin = torch.min(x)
        x = (x - xmin)/(xmax-xmin)
        
        # reshape to the conventional shape for cnn in pytorch
        x = x.reshape(nb_samples,nb_channels,nb_frames,nb_bins)
        
        # encoder path
        print(x.shape)
        conv1 = self.bn1(F.leaky_relu(self.conv1(x),0.2))
        print(conv1.shape)
        
        conv2 = self.bn2(F.leaky_relu(self.conv2(conv1),0.2))
        print(conv2.shape)
        
        conv3 = self.bn3(F.leaky_relu(self.conv3(conv2),0.2))
        print(conv3.shape)
        
        conv4 = self.bn4(F.leaky_relu(self.conv4(conv3),0.2))
        print(conv4.shape)
        
        conv5 = self.bn5(F.leaky_relu(self.conv5(conv4),0.2))
        print(conv5.shape)
        
        conv6 = self.bn6(F.leaky_relu(self.conv6(conv5),0.2))
        print(conv6.shape)
        """
        # decoder path
        x = F.relu(self.dropout1(self.deconv7(conv6)))
        x = self.bn7(x)
        print(x.shape)
        
        x = torch.cat((x,conv5),1)
        x = F.relu(self.dropout1(self.deconv8(x)))
        x = self.bn8(x)
        print(x.shape)
        
        x = torch.cat((x,conv4),1)
        x = F.relu(self.dropout1(self.deconv9(x)))
        x = self.bn9(x)
        print(x.shape)
        
        x = torch.cat((x,conv3),1)
        x = F.relu(self.dropout1(self.deconv10(x)))
        x = self.bn10(x)
        print(x.shape)
        
        x = torch.cat((x,conv2),1)
        x = F.relu(self.dropout1(self.deconv11(x)))
        x = self.bn11(x)
        print(x.shape)
        
        x = torch.cat((x,conv1),1)
        x = torch.sigmoid(self.dropout1(self.deconv12(x)))
        # no batch normalization on this layer?
        print(x.shape)
        
        x = x * x_original
        """
        return x # return the magnitude spectrogram of the estimated source

unmix = deep_u_net(
        n_fft=1024,
        n_hop=768,
        nb_channels=1,
        sample_rate=8192
    )
mix = (torch.rand(1, 1, 300000)+2)**2

unmix.forward(mix)