import shutil
import torch
import os
import numpy as np
import math

def pad_for_stft(signal, hop_length):
    """
    this function pads the given signal so that all samples are taken into account by the stft
    input and output signal have shape (batch_size, nb_channels, nb_timesteps)
    """
    signal_len = signal.shape[-1]
    
    incomplete_frame_len = signal_len % hop_length
    
    if incomplete_frame_len == 0: # no padding needed
        return signal, 0
    else:
        pad_length = hop_length - incomplete_frame_len
        signal = torch.nn.functional.pad(signal, pad=(0, pad_length))
        return signal,pad_length

def memory_check(comment):
    print(comment,torch.cuda.memory_reserved(0)*1e-9, "GB out of",torch.cuda.get_device_properties(0).total_memory*1e-9, "GB used.")

# Returns true if no time steps left over in a Conv1d.
def checkValidConvolution(input_size,kernel_size,stride=1,padding=0,dilation=1,note=""):
    print(note,((input_size + 2*padding - dilation * (kernel_size - 1) - 1)/stride
            + 1).is_integer())

def valid_length(input_size,kernel_size,stride=1,padding=0,dilation=1):
    """
    Return the nearest valid length to use with the model so that
    there is no time steps left over in a a 1dConv, e.g. for all
    layers, size of the (input - kernel_size) % stride = 0.
    """
    #length = math.ceil((input_size + 2*padding - dilation * (kernel_size - 1) - 1)/stride) + 1
    length = math.ceil(torch.true_divide(input_size + 2*padding - dilation * (kernel_size - 1) - 1,stride).item()) + 1
    length = (length - 1) * stride - 2*padding + dilation * (kernel_size - 1) + 1

    return int(length)


def _sndfile_available():
    try:
        import soundfile
    except ImportError:
        return False

    return True


def _torchaudio_available():
    try:
        import torchaudio
    except ImportError:
        return False

    return True


def get_loading_backend():
    if _torchaudio_available():
        return torchaudio_loader

    if _sndfile_available():
        return soundfile_loader


def get_info_backend():
    if _torchaudio_available():
        return torchaudio_info

    if _sndfile_available():
        return soundfile_info


def soundfile_info(path):
    import soundfile
    info = {}
    sfi = soundfile.info(path)
    info['samplerate'] = sfi.samplerate
    info['samples'] = int(sfi.duration * sfi.samplerate)
    info['duration'] = sfi.duration
    return info


def soundfile_loader(path, start=0, dur=None):
    import soundfile
    # get metadata
    info = soundfile_info(path)
    start = int(start * info['samplerate'])
    # check if dur is none
    if dur:
        # stop in soundfile is calc in samples, not seconds
        stop = start + int(dur * info['samplerate'])
    else:
        # set to None for reading complete file
        stop = dur

    audio, _ = soundfile.read(
        path,
        always_2d=True,
        start=start,
        stop=stop
    )
    return torch.FloatTensor(audio.T)


def torchaudio_info(path):
    import torchaudio
    # get length of file in samples
    info = {}
    si, _ = torchaudio.info(str(path))
    info['samplerate'] = si.rate
    info['samples'] = si.length // si.channels
    info['duration'] = info['samples'] / si.rate
    return info


def torchaudio_loader(path, start=0, dur=None):
    import torchaudio
    info = torchaudio_info(path)
    # loads the full track duration
    if dur is None:
        sig, rate = torchaudio.load(path)
        return sig
        # otherwise loads a random excerpt
    else:
        num_frames = int(dur * info['samplerate'])
        offset = int(start * info['samplerate'])
        sig, rate = torchaudio.load(
            path, num_frames=num_frames, offset=offset
        )
        return sig


def load_info(path):
    loader = get_info_backend()
    return loader(path)


def load_audio(path, start=0, dur=None):
    loader = get_loading_backend()
    return loader(path, start=start, dur=dur)


def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    freqs = np.linspace(
        0, float(rate) / 2, n_fft // 2 + 1,
        endpoint=True
    )
    #print(freqs[1480:1490])
    #print(np.where(freqs <= bandwidth)[0])
    return np.max(np.where(freqs <= bandwidth)[0]) + 1


def save_checkpoint(
    state, is_best, path, target
):
    # save full checkpoint including optimizer
    torch.save(
        state,
        os.path.join(path, target + '.chkpnt')
    )
    if is_best:
        # save just the weights
        torch.save(
            state['state_dict'],
            os.path.join(path, target + '.pth')
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta