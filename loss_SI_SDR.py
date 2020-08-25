import torch

def sisdr(estimates, targets, eps=1e-8):
    """
    calculate training loss
    input:
          estimates: separated signals, (batch_size,1,nb_samples) tensor
          targets: reference signals, (batch_size,1,nb_samples) tensor
    Return:
          sisdr: SI-SDR mean over all samples in a batch
    """

    if estimates.shape != targets.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                estimates.shape, targets.shape))
    
    #x_zeroMean = x - torch.mean(x, dim=-1, keepdim=True)
    #s_zeroMean = s - torch.mean(s, dim=-1, keepdim=True)
    
    # scaling [batch_size,1,1]
    scaling = torch.sum(estimates * targets, dim=-1,keepdim=True) / (torch.sum(targets * targets, dim=-1, keepdim=True) + eps) # to discuss
    # [batch_size,1,nb_samples]
    e_target = scaling * targets
    
    #e_target = targets
    e_residual = estimates - e_target
    
    # [batch_size,1,1]
    Starg= torch.sum(e_target**2,dim=-1)
    Sres= torch.sum(e_residual**2,dim=-1)

    # [batch_size,1,1]
    SI_SDR = - 10*torch.log10(Starg/(eps+Sres) + eps)

    return torch.mean(SI_SDR) # return mean over all samples in a batch

if __name__ == '__main__':
    import torchaudio
    import museval
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mixture, sample_rate = torchaudio.load('convtasnet_1_0mixture.wav')
    target, sample_rate = torchaudio.load('convtasnet_1_0target.wav')
    estimate,sample_rate = torchaudio.load('convtasnet_100_0estimate.wav')
    
    print("Original SI-SNR:",sisnr(mixture,target))
    print("New SI-SNR:",sisnr(estimate,target))
    print(museval.evaluate(target,estimate))
    print(museval.evaluate(target,mixture))