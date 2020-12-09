import torch
import math
import numpy as np

def sdr(estimates, targets,eps=1e-8,scale_invariant=True):
    """
    Input:
        estimates: torch.tensor or numpy.array
            separated signals, of shape (batch_size,nb_channels,nb_samples)
            OR (nb_channels,nb_samples) OR (nb_samples) tensor
        targets: torch.tensor or numpy.array
            reference signals, (batch_size,nb_channels,nb_samples) 
            OR (nb_channels,nb_samples) OR (nb_samples) tensor
        eps : float
            a small value that may be set to avoid divison by 0 and log(0)
        scale_invariant : boolean
            decide between the scale invariant version SI-SDR, or traditional SDR
    Return:
        - If scale_invariant=True : returns SI-SDR computed over each batch, channel
                                    that is then averaged over batch and channels.
        
        - If scale_invariant=False : returns SDR computed over each batch, channel
                                    that is then averaged over batch and channels.
    """
    
    if type(estimates) == np.ndarray:
        estimates = torch.from_numpy(estimates)
        targets = torch.from_numpy(targets)
    
    if estimates.shape != targets.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                estimates.shape, targets.shape))
        
    if len(estimates.shape) == 1: # add batch and channel dimension
        estimates = estimates[None,None,...]
        targets = targets[None,None,...]
    
    if len(estimates.shape) == 2: # add batch dimension
        estimates = estimates[None,...]
        targets = targets[None,...]
    
    batch_size,nb_channels,nb_samples = estimates.shape
    
    # reshaped [batch_size,nb_channels,1, nb of samples]
    estimates_reshaped = estimates.view(batch_size,nb_channels,1,-1)
    targets_reshaped = targets.view(batch_size,nb_channels,1,-1)
    
    if scale_invariant == True: # SI-SDR case
        # scaling [batch_size,nb_channels,1,1]
        scaling = torch.sum(estimates_reshaped * targets_reshaped, dim=-1,keepdim=True) / (torch.sum(targets_reshaped * targets_reshaped, dim=-1, keepdim=True) + eps)
        e_target = scaling * targets_reshaped
    else:
        e_target = targets_reshaped
    # e_target [batch_size,nb_channels,1,nb of samples]
        
    e_residual = estimates_reshaped - e_target

    # Starg [batch_size,nb_channels,1]
    Starg= torch.sum(e_target**2,dim=-1,keepdim=True).view(batch_size,nb_channels,-1)
    Sres= torch.sum(e_residual**2,dim=-1,keepdim=True).view(batch_size,nb_channels,-1)
        
    # SI_SDR [batch_size,nb_channels,1]
    SI_SDR = 10*torch.log10(Starg/(eps+Sres) + eps)
    
    return torch.mean(SI_SDR).item() # returns a float!

def sisdr_framewise(estimates, targets, sample_rate,eps=1e-8,scale_invariant=True):
    """
    Input:
        estimates: torch.tensor
            separated signals, of shape (batch_size,nb_channels,nb_samples) or
            (nb_channels,nb_samples)
        targets: torch.tensor
            reference signals, of shape (batch_size,nb_channels,nb_samples) or
            (nb_channels,nb_samples)
        sample_rate: int
            sample rate of the estimates and targets
        eps : float
            a small value that may be set to avoid divison by 0 and log(0)
        scale_invariant : boolean
            decide between the scale invariant version SI-SDR, or traditional SDR
    Return:
        SI-SDR over each seconds for each batch and channels, i.e. of shape 
        (batch_size,nb_channels,number of seconds)
    """

    if estimates.shape != targets.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                estimates.shape, targets.shape))
    
    #x_zeroMean = x - torch.mean(x, dim=-1, keepdim=True)
    #s_zeroMean = s - torch.mean(s, dim=-1, keepdim=True)
    
    if len(estimates.shape) == 2: # add batch dimension
        estimates = estimates[None,...]
        targets = targets[None,...]
    
    batch_size,nb_channels,nb_samples = estimates.shape
    
    # Discard the end of the signals of less than 1s, and reshaped so that to compute
    # SI-SDR on 1s portions
    # reshaped [batch_size,nb_channels,number of seconds, sample rate]
    estimates_reshaped = estimates[...,:nb_samples//sample_rate * sample_rate].view(batch_size,nb_channels,-1,sample_rate)
    targets_reshaped = targets[...,:nb_samples//sample_rate * sample_rate].view(batch_size,nb_channels,-1,sample_rate)
    
    if scale_invariant == True:
        # scaling [batch_size,nb_channels,number of seconds,1]
        scaling = torch.sum(estimates_reshaped * targets_reshaped, dim=-1,keepdim=True) / (torch.sum(targets_reshaped * targets_reshaped, dim=-1, keepdim=True) + eps)
        e_target = scaling * targets_reshaped
    else:
        e_target = targets_reshaped
    # e_target [batch_size,1,number of seconds,sample rate]
    
    e_residual = estimates_reshaped - e_target
        
    # Starg [batch_size,number of seconds,1]
    Starg= torch.sum(e_target**2,dim=-1,keepdim=True).view(batch_size,nb_channels,-1)
    Sres= torch.sum(e_residual**2,dim=-1,keepdim=True).view(batch_size,nb_channels,-1)
    
    # SI_SDR [batch_size,nb_channels,number of seconds]
    SI_SDR = - 10*torch.log10(Starg/(eps+Sres) + eps)

    return SI_SDR

def ideal_SDR_framewise(estimates, targets, sample_rate):
    """
    Input:
        estimates: torch.tensor
            separated signals, of shape (batch_size,nb_channels,nb_samples) or
            (nb_channels,nb_samples)
        targets: torch.tensor
            reference signals, of shape (batch_size,nb_channels,nb_samples) or
            (nb_channels,nb_samples)
        sample_rate: int
            sample rate of the estimates and targets
    Return:
        SDR over each seconds for each batch and channels, i.e. of shape 
        (batch_size,nb_channels,number of seconds). A scaling has been applied
        to the estimates, so as to obtain the highest SDR value to make this 
        metric unsensitive to scale
    """
    
    torch_factor_ideal = torch.sum(estimates*targets,dim=-1)/torch.sum(estimates*estimates,dim=-1)
    torch_factor_ideal = torch.unsqueeze(torch_factor_ideal,dim=-1)
    
    torch_estimate_ideal = torch_factor_ideal * estimates

    return sisdr_framewise(torch_estimate_ideal, targets,
                        sample_rate,scale_invariant=False,eps=0)

def ideal_SDR(estimates, targets):
    """
    Input:
        estimates: torch.tensor
            separated signals, of shape (batch_size,nb_channels,nb_samples) or
            (nb_channels,nb_samples) or (nb_samples)
        targets: torch.tensor
            reference signals, of shape (batch_size,nb_channels,nb_samples) or
            (nb_channels,nb_samples) or (nb_samples)
    Return:
        SDR computed over each batch, channel that is then averaged over batch
        and channels. A scaling has been applied to the estimates, so as to 
        obtain the highest SDR value to make this metric unsensitive to scale
    """
    
    torch_factor_ideal = torch.sum(estimates*targets,dim=-1)/torch.sum(estimates*estimates,dim=-1)
    torch_factor_ideal = torch.unsqueeze(torch_factor_ideal,dim=-1)
    
    torch_estimate_ideal = torch_factor_ideal * estimates

    return sdr(torch_estimate_ideal, targets,scale_invariant=False,eps=0)