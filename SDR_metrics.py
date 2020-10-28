import torch
import math
import numpy as np

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
    torch_factor_ideal = torch.sum(estimates*targets,dim=-1)/torch.sum(estimates*estimates,dim=-1)
    torch_factor_ideal = torch.unsqueeze(torch_factor_ideal,dim=-1)
    
    torch_estimate_ideal = torch_factor_ideal * estimates

    return sisdr_framewise(torch_estimate_ideal, targets,
                        sample_rate,scale_invariant=False,eps=0)
    
def loss_SI_SDR(SI_SDR_framewise,eps=1e-8):
    if eps == 0:
        SI_SDR_framewise = SI_SDR_framewise[torch.isfinite(SI_SDR_framewise)]
    # return mean over all samples in a batch and channels
    return torch.mean(SI_SDR_framewise) 

def metric_SI_SDR(SI_SDR_framewise,eps=1e-8):
    if eps == 0:
        SI_SDR_finite = SI_SDR_framewise[torch.isfinite(SI_SDR_framewise)]
    else:
        SI_SDR_finite = SI_SDR_framewise
    
    # return mean over all samples in a batch and channels
    return np.median(SI_SDR_finite.numpy()) 

if __name__ == '__main__':
    import torchaudio
    import museval
    import numpy as np
    import mir_eval
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    target, sample_rate = torchaudio.load('/tsi/doctorants/fmarty/Datasets/MUSDB18_16000wav/test/AM Contra - Heart Peripheral/vocals.wav')
    estimate,sample_rate = torchaudio.load('/tsi/doctorants/fmarty/executedJobs/09-14_CTNbaselineJoint/outDir/test/AM Contra - Heart Peripheral/vocals.wav')    

    target = np.array(target)[:,10*16000:40*16000]
    estimate = np.array(estimate)[:,10*16000:40*16000]
    
    #target = target[None,...]
    #estimate = estimate[None,...]
    
    #print(target.shape)
    sr = 16000
    
    #torch.manual_seed(7)
    
    #estimate = 4*torch.rand(1, 24*44100)-2
    #target = 4*torch.rand(1, 24*44100)-2
    
    #estimate = estimate.numpy()
    #target = target.numpy()
    
    """
    estimate = np.zeros((1,60*44100))
    target = np.zeros((1,60*44100))
    
    for i in range(60*44100):
        estimate[0][i] = 0.4*np.sin(i*0.1)+0.2
        target[0][i] = estimate[0][i] + 5*random.random()
    """
        
    estimate_1 = estimate
    estimate_0_5 = 0.5*estimate
    estimate_4 = 4*estimate
        
    
    sdr_1_museval = museval.evaluate(target,estimate_1,win=sr, hop=sr)[0]
    sdr_0_5_museval = museval.evaluate(target,estimate_0_5,
                        win=sr, hop=sr)[0]
    sdr_4_museval = museval.evaluate(target,estimate_4,win=sr, hop=sr)[0]
    
    print("SDR museval.evaluate (estimate*1):",np.median(sdr_1_museval))
    print("SDR museval.evaluate (estimate*0.5):",np.median(sdr_0_5_museval))
    print("SDR museval.evaluate (estimate*4):",np.median(sdr_4_museval))
    print("----")
    
    sdr_1_museval_v3 = museval.evaluate(target,estimate_1,
                        win=sr, hop=sr,mode='v3')[0]
    sdr_0_5_museval_v3 = museval.evaluate(target,estimate_0_5,
                        win=sr, hop=sr,mode='v3')[0]
    sdr_4_museval_v3 = museval.evaluate(target,estimate_4,
                        win=sr, hop=sr,mode='v3')[0]
    
    print("SDR museval.evaluate v3 (estimate*1):",np.median(sdr_1_museval_v3))
    print("SDR museval.evaluate v3 (estimate*0.5):",np.median(sdr_0_5_museval_v3))
    print("SDR museval.evaluate v3 (estimate*4):",np.median(sdr_4_museval_v3))
    print("----")

    sdr_1_mir = mir_eval.separation.bss_eval_images_framewise(
                            target, estimate_1, window=sr, hop=sr)[0]

    
    sdr_0_5_mir = mir_eval.separation.bss_eval_images_framewise(
                                target, estimate_0_5, window=sr, hop=sr)[0]
    
    sdr_4_mir = mir_eval.separation.bss_eval_images_framewise(
                                target, estimate_4, window=sr, hop=sr)[0]

    print("SDR mir_eval_image (estimate*1):",np.median(sdr_1_mir))
    print("SDR mir_eval_image (estimate*0.5):",np.median(sdr_0_5_mir))
    print("SDR mir_eval_image (estimate*4):",np.median(sdr_4_mir))
    print("----")
    
    sdr_1_mir = mir_eval.separation.bss_eval_sources_framewise(
                            target, estimate_1, window=sr, hop=sr)[0]

    
    sdr_0_5_mir = mir_eval.separation.bss_eval_sources_framewise(
                                target, estimate_0_5, window=sr, hop=sr)[0]
    
    sdr_4_mir = mir_eval.separation.bss_eval_sources_framewise(
                                target, estimate_4, window=sr, hop=sr)[0]
    
    print("SDR mir_eval_source (estimate*1):",np.median(sdr_1_mir))
    print("SDR mir_eval_source (estimate*0.5):",np.median(sdr_0_5_mir))
    print("SDR mir_eval_source (estimate*4):",np.median(sdr_4_mir))
    print("----")
    
    torch_estimate_1 = torch.from_numpy(estimate_1)
    torch_estimate_0_5 = torch.from_numpy(estimate_0_5)
    torch_estimate_4 = torch.from_numpy(estimate_4)
    torch_target = torch.from_numpy(target)
    torch.set_printoptions(precision=10)
    
    sdr_1_mine = -metric_SI_SDR(sisdr_framewise(torch_estimate_1, torch_target,
                        sr,scale_invariant=False,eps=0),eps=0)
    sdr_0_5_mine = -metric_SI_SDR(sisdr_framewise(torch_estimate_0_5,
                        torch_target, sr,scale_invariant=False,eps=0),eps=0)
    sdr_4_mine = -metric_SI_SDR(sisdr_framewise(torch_estimate_4, torch_target,
                        sr,scale_invariant=False,eps=0),eps=0)
    sdr_ideal_mine = -metric_SI_SDR(ideal_SDR_framewise(torch_estimate_1, torch_target, sr),eps=0)
    
    print("SDR mine (estimate*1):",sdr_1_mine)
    print("SDR mine (estimate*0.5):",sdr_0_5_mine)
    print("SDR mine (estimate*4):",sdr_4_mine)
    print("SDR mine (ideal multiplication factor):",sdr_ideal_mine)
    print("----")
    
    si_sdr_1_mine = -metric_SI_SDR(sisdr_framewise(torch_estimate_1, torch_target,
                        sr,scale_invariant=True,eps=0),eps=0)
    si_sdr_0_5_mine = -metric_SI_SDR(sisdr_framewise(torch_estimate_0_5,
                        torch_target, sr,scale_invariant=True,eps=0),eps=0)
    si_sdr_4_mine = -metric_SI_SDR(sisdr_framewise(torch_estimate_4, torch_target,
                        sr,scale_invariant=True,eps=0),eps=0)

    print("SI-SDR mine (estimate*1):",si_sdr_1_mine)
    print("SI-SDR mine (estimate*0.5):",si_sdr_0_5_mine)
    print("SI-SDR mine (estimate*4):",si_sdr_4_mine)
    
    
    
    target_44100, sample_rate_1 = torchaudio.load('/tsi/doctorants/fmarty/Datasets/MUSDB18wav/train/Music Delta - Britpop/vocals.wav')
    mixture_44100,sample_rate_1 = torchaudio.load('/tsi/doctorants/fmarty/Datasets/MUSDB18wav/train/Music Delta - Britpop/mixture.wav')
    
    target_16000,sample_rate_2 = torchaudio.load('/tsi/doctorants/fmarty/Datasets/MUSDB18_16000wav/train/Music Delta - Britpop/vocals.wav')
    mixture_16000,sample_rate_2 = torchaudio.load('/tsi/doctorants/fmarty/Datasets/MUSDB18_16000wav/train/Music Delta - Britpop/mixture.wav')
    
    si_sdr_44100_mine = -metric_SI_SDR(sisdr_framewise(mixture_44100, target_44100,
                        sample_rate_1,scale_invariant=True,eps=0),eps=0)
                        
    si_sdr_16000_mine = -metric_SI_SDR(sisdr_framewise(mixture_16000, target_16000,
                        sample_rate_2,scale_invariant=True,eps=0),eps=0)
    
    print("--- An other song at different sampling rate, what is the influence? ---")
    print("SI-SDR at 44100 Hz:",si_sdr_44100_mine)
    print("SI-SDR at 16000 Hz:",si_sdr_16000_mine)