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
    import numpy as np
    import mir_eval
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    mixture, sample_rate = torchaudio.load('convtasnet_1_0mixture.wav')
    target, sample_rate = torchaudio.load('convtasnet_1_0target.wav')
    estimate,sample_rate = torchaudio.load('convtasnet_300_0estimate.wav')    

    target = np.array(target)
    estimate = np.array(estimate)
    
    
    sr = 44100
    
    #torch.manual_seed(7)
    
    #estimate = 4*torch.rand(1, 12*44100)-2
    #target = 4*torch.rand(1, 12*44100)-2
    
    """
    estimate = np.zeros((1,12*44100))
    target = np.zeros((1,12*44100))
    
    for i in range(12*44100):
        estimate[0][i] = 0.4*np.sin(i*0.1)+0.2
        target[0][i] = np.sin(i*0.1)
    """
    #scaling = torch.sum(estimate * target, dim=-1,keepdim=True) / (torch.sum(target * target, dim=-1, keepdim=True))
    #target = scaling * target
    
    #print("-------Original SI-SNR:",sisdr(estimate,target))
    #print("-------New SI-SNR:",sisdr_framewise(estimate,target,sample_rate=sr))
    
    estimate_1 = estimate
    estimate_0_5 = 0.5*estimate
    estimate_4 = 4*estimate
        
    sdr_1_museval = museval.evaluate(target,estimate_1,win=sr, hop=sr)[0]
    sdr_0_5_museval = museval.evaluate(target,estimate_0_5,
                        win=sr, hop=sr)[0]
    sdr_4_museval = museval.evaluate(target,estimate_4,win=sr, hop=sr)[0]
    
    print("SDR museval estimate*1:",np.median(sdr_1_museval))
    print("SDR museval estimate*0.5:",np.median(sdr_0_5_museval))
    print("SDR museval estimate*4:",np.median(sdr_4_museval))
    print("----")
    sdr_1_museval_v3 = museval.evaluate(target,estimate_1,
                        win=sr, hop=sr,mode='v3')[0]
    sdr_0_5_museval_v3 = museval.evaluate(target,estimate_0_5,
                        win=sr, hop=sr,mode='v3')[0]
    sdr_4_museval_v3 = museval.evaluate(target,estimate_4,
                        win=sr, hop=sr,mode='v3')[0]
    
    print("SDR museval v3 estimate*1:",np.median(sdr_1_museval_v3))
    print("SDR museval v3 estimate*0.5:",np.median(sdr_0_5_museval_v3))
    print("SDR museval v3 estimate*4:",np.median(sdr_4_museval_v3))
    print("----")

    sdr_1_mir = mir_eval.separation.bss_eval_images_framewise(
                            target, estimate_1, window=sr, hop=sr)[0]

    
    sdr_0_5_mir = mir_eval.separation.bss_eval_images_framewise(
                                target, estimate_0_5, window=sr, hop=sr)[0]
    
    sdr_4_mir = mir_eval.separation.bss_eval_images_framewise(
                                target, estimate_4, window=sr, hop=sr)[0]

    print("SDR estimate*1 mir_eval_image:",np.median(sdr_1_mir))
    print("SDR estimate*0.5 mir_eval_image:",np.median(sdr_0_5_mir))
    print("SDR estimate*4 mir_eval_image:",np.median(sdr_4_mir))
    print("----")
    
    sdr_1_mir = mir_eval.separation.bss_eval_sources_framewise(
                            target, estimate_1, window=sr, hop=sr)[0]

    
    sdr_0_5_mir = mir_eval.separation.bss_eval_sources_framewise(
                                target, estimate_0_5, window=sr, hop=sr)[0]
    
    sdr_4_mir = mir_eval.separation.bss_eval_sources_framewise(
                                target, estimate_4, window=sr, hop=sr)[0]
    
    print("SDR estimate*1 mir_eval_source:",np.median(sdr_1_mir))
    print("SDR estimate*0.5 mir_eval_source:",np.median(sdr_0_5_mir))
    print("SDR estimate*4 mir_eval_source:",np.median(sdr_4_mir))
    print("----")
    
    torch_estimate_1 = torch.from_numpy(estimate_1)
    torch_estimate_0_5 = torch.from_numpy(estimate_0_5)
    torch_estimate_4 = torch.from_numpy(estimate_4)
    torch_target = torch.from_numpy(target)
    
    
    sdr_1_mine = sisdr_framewise(torch_estimate_1, torch_target,
                        sr,scale_invariant=False)
    sdr_0_5_mine = sisdr_framewise(torch_estimate_0_5,
                        torch_target, sr,scale_invariant=False)
    sdr_4_mine = sisdr_framewise(torch_estimate_4, torch_target,
                        sr,scale_invariant=False)
    torch.set_printoptions(precision=10)
    print("SDR estimate*1 mine:",sdr_1_mine)
    print("SDR estimate*0.5 mine:",sdr_0_5_mine)
    print("SDR estimate*4 mine:",sdr_4_mine)
    print("----")
    
    si_sdr_1_mine = sisdr_framewise(torch_estimate_1, torch_target,
                        sr,scale_invariant=True)
    si_sdr_0_5_mine = sisdr_framewise(torch_estimate_0_5,
                        torch_target, sr,scale_invariant=True)
    si_sdr_4_mine = sisdr_framewise(torch_estimate_4, torch_target,
                        sr,scale_invariant=True)

    print("SDR estimate*1 mine:",si_sdr_1_mine)
    print("SDR estimate*0.5 mine:",si_sdr_0_5_mine)
    print("SDR estimate*4 mine:",si_sdr_4_mine)

    