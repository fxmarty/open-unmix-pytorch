import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self,
                normalization_style,
                input_mean=None,
                input_scale=None,
                nb_bins=None
    ):
        super(Normalize,self).__init__()
        
        self.normalization_style = normalization_style
        
        if input_mean is not None:
            input_mean = torch.from_numpy(
                -input_mean[:nb_bins]
            ).float()
        else:
            input_mean = torch.zeros(nb_bins)
        
        if input_scale is not None:
            input_scale = torch.from_numpy(
                1.0/input_scale[:nb_bins]
            ).float()
        else:
            input_scale = torch.ones(nb_bins)

        self.input_mean = nn.Parameter(input_mean)
        self.input_scale = nn.Parameter(input_scale)
    
    def forward(self,x):
        if self.normalization_style == "overall":
            x += self.input_mean
            x *= self.input_scale
        
        if self.normalization_style == "batch-specific":
            xmax = torch.max(x)
            xmin = torch.min(x)
            x = (x - xmin)/(xmax-xmin)
        return x