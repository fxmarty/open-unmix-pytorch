import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self,
                normalization_style,
                input_mean=None,
                input_scale=None,
                nb_total_bins=None,
                imprint=True
    ):
        super(Normalize,self).__init__()
        
        self.normalization_style = normalization_style
        
        if imprint == True:
            print("Normalization set to \"" + self.normalization_style + "\".")
        
        if normalization_style == 'overall':
            if input_mean is not None:
                input_mean = torch.from_numpy(
                    -input_mean[:nb_total_bins]
                ).float()
            else:
                input_mean = torch.zeros(nb_total_bins)
            
            if input_scale is not None:
                input_scale = torch.from_numpy(
                    1.0/input_scale[:nb_total_bins]
                ).float()
            else:
                input_scale = torch.ones(nb_total_bins)
    
            self.input_mean = nn.Parameter(input_mean)
            self.input_scale = nn.Parameter(input_scale)
    
    def forward(self,x):
        if self.normalization_style == 'overall':
            x += self.input_mean
            x *= self.input_scale
        
        if self.normalization_style == 'batch-specific':
            xmax = torch.max(x)
            xmin = torch.min(x)
            x = (x - xmin)/(xmax-xmin)
        
        if self.normalization_style == 'none':
            pass
            
        return x