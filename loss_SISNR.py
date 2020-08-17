import torch
from itertools import permutations

def sisnr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zeroMean = x - torch.mean(x, dim=-1, keepdim=True)
    s_zeroMean = s - torch.mean(s, dim=-1, keepdim=True)
    
    
    s_target = torch.sum(x_zeroMean * s_zeroMean, dim=-1,keepdim=True) * s_zeroMean / (torch.norm(s_zeroMean, dim=-1, keepdim=True)**2 + eps)
    
    e_noise = x_zeroMean - s_target
    return sum(2*10*torch.log10(eps + torch.norm(s_target) / (torch.norm(e_noise) + eps)))