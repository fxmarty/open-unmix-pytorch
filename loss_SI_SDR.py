import torch
import math
import numpy as np

def sisdr(estimates, targets,eps=0,scale_invariant=True):
    """
    calculate training loss
    input:
          estimates: separated signals, (batch_size,nb_channels,nb_samples) tensor
          targets: reference signals, (batch_size,nb_channels,nb_samples) tensor
    Return:
          sisdr: SI-SDR mean over all samples in a batch
    """

    if estimates.shape != targets.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                estimates.shape, targets.shape))
        
    if len(estimates.shape) == 2: # add batch dimension
        estimates = estimates[None,...]
        targets = targets[None,...]
    
    if scale_invariant == True:
        # scaling [batch_size,nb_channels,1]
        scaling = torch.sum(estimates * targets, dim=-1,keepdim=True) / (torch.sum(targets * targets, dim=-1, keepdim=True) + eps) # to discuss
    else:
        scaling = 1
    
    # e_target [batch_size,nb_channels,nb_samples]
    e_target = scaling * targets
    
    e_residual = estimates - e_target
    
    # Starg [batch_size,nb_channels,1]
    Starg= torch.sum(e_target**2,dim=-1,keepdim=True)
    Sres= torch.sum(e_residual**2,dim=-1,keepdim=True)
    
    # SI_SDR [batch_size,nb_channels,1]
    SI_SDR = - 10*torch.log10(Starg/(eps+Sres) + eps)

    return torch.mean(SI_SDR) # return mean over all samples in a batch

def sisdr_framewise(estimates, targets, sample_rate,eps=1e-8,scale_invariant=True):
    """
    input:
          estimates: separated signals, (batch_size,nb_channels,nb_samples)
                        OR (nb_channels,nb_samples) tensor
          targets: reference signals, (batch_size,nb_channels,nb_samples) 
                        OR (nb_channels,nb_samples) tensor tensor
          sample_rate: sample rate of the estimates and targets
    Return:
          sisdr: SI-SDR mean over all samples in a batch
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
        scaling = torch.sum(estimates_reshaped * targets_reshaped, dim=-1,keepdim=True) / (torch.sum(targets_reshaped * targets_reshaped, dim=-1, keepdim=True) + eps) # to discuss
    else:
        scaling = 1

    # e_target [batch_size,1,number of seconds,sample rate]
    e_target = scaling * targets_reshaped
    
    #e_target = targets
    e_residual = estimates_reshaped - e_target
    
    # Starg [batch_size,number of seconds,1]
    Starg= torch.sum(e_target**2,dim=-1,keepdim=True).view(batch_size,nb_channels,-1)
    Sres= torch.sum(e_residual**2,dim=-1,keepdim=True).view(batch_size,nb_channels,-1)
    
    # SI_SDR [batch_size,nb_channels,number of seconds]
    SI_SDR = - 10*torch.log10(Starg/(eps+Sres) + eps)
    
    if eps == 0:
        SI_SDR = SI_SDR[torch.isfinite(SI_SDR)]
    
    return torch.mean(SI_SDR) # return mean over all samples in a batch and channels


if __name__ == '__main__':
    import torchaudio
    import museval
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    """
    torch.manual_seed(42)
    estimate = torch.rand(1, 1, 12*44100)
    target = torch.rand(1, 1, 12*44100)
    
    print(sisdrNew(target,estimate,44100))
    print(np.mean(museval.evaluate(target[0],estimate[0])[0]))
    """
    
    mixture, sample_rate = torchaudio.load('convtasnet_1_0mixture.wav')
    target, sample_rate = torchaudio.load('convtasnet_1_0target.wav')
    estimate,sample_rate = torchaudio.load('convtasnet_400_0estimate.wav')
    
    mixture = mixture[...,:44100]
    target = target[...,:44100]
    estimate = estimate[...,:44100]
    
    print("-------Original SI-SNR:",sisdr_framewise(estimate,target,44100))
    print("-------New SI-SNR:",sisdr_framewise(estimate,target,44100))
    print(museval.evaluate(target,estimate))    
