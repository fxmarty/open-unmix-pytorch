import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_model_summary
from torchsummary import summary

class Encoder(nn.Module):
    def __init__(
        self,
        number_of_phonemes,
        bottleneck_size
    ):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Sequential(
                        nn.Linear(number_of_phonemes,32,bias=False),
                        nn.ReLU()
                )
        
        self.fc2 = nn.Sequential(
                        nn.Linear(32,16,bias=False),
                        nn.ReLU()
                )
        
        self.fc3 = nn.Sequential(
                        nn.Linear(16,bottleneck_size,bias=False),
                        nn.ReLU(),
                )
        
        """
        self.encoding_lstm = nn.LSTM(
                                    input_size=number_of_phonemes,
                                    hidden_size=bottleneck_size//2,
                                    num_layers=2,
                                    batch_first=True,
                                    bidirectional=True
                            )
        """
        
    def forward(self,phoneme):
        x = self.fc1(phoneme)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
        #return self.encoding_lstm(phoneme)[0]

class Decoder(nn.Module):
    def __init__(
        self,
        number_of_phonemes,
        bottleneck_size
    ):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Sequential(
                        nn.Linear(bottleneck_size,16,bias=False),
                        nn.ReLU()
                )
        
        self.fc2 = nn.Sequential(
                        nn.Linear(16,32,bias=False),
                        nn.ReLU()
                )
        
        self.fc3 = nn.Sequential(
                        nn.Linear(32,number_of_phonemes,bias=False),
                        nn.Sigmoid()
                )
        
    def forward(self,phoneme_encoded):
        x = self.fc1(phoneme_encoded)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class Autoencoder(nn.Module):
    def __init__(
        self,
        number_of_phonemes = 65,
        bottleneck_size=8
    ):

        super(Autoencoder, self).__init__()
        
        self.encoder = Encoder(number_of_phonemes,bottleneck_size)
        self.decoder = Decoder(number_of_phonemes,bottleneck_size)
        
    def forward(self, phoneme):
        phoneme_encoded = self.encoder(phoneme)

        phoneme_estimated = self.decoder(phoneme_encoded)
        
        return phoneme_estimated

if __name__ == '__main__':
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(
        number_of_phonemes=64
        ).to(device)
        
    data = torch.zeros((8,4789,64)).to(device)
    print(pytorch_model_summary.summary(model, data, show_input=False))
    
    print(model.decoder)
    
    summary(model, input_data=torch.zeros((16,4789,64)),batch_dim=0)