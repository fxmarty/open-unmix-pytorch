import torch
import numpy as np
import argparse
import soundfile as sf
import norbert
import json
from pathlib import Path
import scipy.signal
import resampy

from models import open_unmix

import utils
import warnings
import tqdm
from contextlib import redirect_stderr
import io

def load_model(target, model_name='umxhq', device='cpu'):
    """
    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)
    """
    model_path = Path(model_name).expanduser()

    if not model_path.exists():
        raise NameError('Model path is wrong')
            # assume model is a path to a local model_name directory
    else:
        # load model from disk
        with open(Path(model_path, target + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = next(Path(model_path).glob("%s*.pth" % target))
        state = torch.load(
            target_model_path,
            map_location=device
        )
        
        max_bin = utils.bandwidth_to_max_bin(
            state['sample_rate'],
            results['args']['nfft'],
            results['args']['bandwidth']
        ) # returns the number of frequency bins so that their frequency is lower
        # than the bandwidth indicated in the .json files

        if results['args']['modelname'] == 'open-unmix':
            unmix = open_unmix.OpenUnmix(
                normalization_style=results['args']['normalization_style'],
                n_fft=results['args']['nfft'],
                n_hop=results['args']['nhop'],
                nb_channels=results['args']['nb_channels'],
                hidden_size=results['args']['hidden_size'],
                max_bin=max_bin,
                single_phoneme=results['args']['single_phoneme']
            )
            unmix.stft.center = True
            unmix.phoneme_network.center = True
                    
        if results['args']['modelname'] == 'deep-u-net':
            unmix = deep_u_net.Deep_u_net(
                normalization_style=results['args']['normalization_style'],
                n_fft=results['args']['nfft'],
                n_hop=results['args']['nhop'],
                nb_channels=results['args']['nb_channels'],
                single_phoneme=results['args']['single_phoneme']
            )
            unmix.stft.center = True
            unmix.phoneme_network.center = True
        
        if results['args']['modelname'] == 'convtasnet':
            unmix = convtasnet.ConvTasNet(
                normalization_style=results['args']['normalization_style'],
                nb_channels=results['args']['nb_channels'],
                sample_rate=16000,
                C=2 if results['args']['joint'] else 1,
                single_phoneme=results['args']['single_phoneme']
            )
            
            
        unmix.load_state_dict(state) # Load saved model
        unmix.eval()
        unmix.to(device)
            
        return unmix,results


def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio


def separate(
    audio,
    phoneme,
    targets,
    model_name='umxhq',
    niter=1, softmask=False, alpha=1.0,
    residual_model=False, device='cpu',
    offset=0
):
    """
    Performing the separation on audio input

    Parameters
    ----------
    audio: np.ndarray [shape=(nb_samples, nb_channels, nb_timesteps)]
        mixture audio

    targets: list of str
        a list of the separation targets.
        Note that for each target a separate model is expected
        to be loaded.

    model_name: str
        name of torchhub model or path to model folder, defaults to `umxhq`

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.

    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False

    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0

    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False

    device: str
        set torch device. Defaults to `cpu`.

    Returns
    -------
    estimates: `dict` [`str`, `np.ndarray`]
        dictionary of all restimates as performed by the separation model.

    """
    with torch.no_grad():
        # convert numpy audio to torch
        mixture = torch.tensor(audio.T[None, ...]).float().to(device)
        # mixture shape [1,2,nb_time_points]
        
        single_phoneme = len(phoneme.shape) == 1
        
        # At the front, we add one a frame of 0 due to the missing frame at the front
        # time_transform_posteriograms.timeTransform is built for this behavior
        # We add some zeros at the end just in case
        if single_phoneme:
            phoneme = torch.cat((torch.zeros(1),phoneme,torch.zeros(5)),dim=0)
        else:
            phoneme = torch.cat((torch.zeros(1,64),phoneme,torch.zeros(5,64)),dim=0)
        
        # add batch dimension
        phoneme = phoneme[None,...].to(device)
        
        source_names = []
        V = []
        
        for j, target in enumerate(targets):
            unmix_target,args = load_model(
                target=target,
                model_name=model_name,
                device=device
            )
            
            modelname = args['args']['modelname']
            nb_channels_model = args['args']['nb_channels']
            
            # If the model takes mono as input,
            # put the channels in the number of samples dim
            if nb_channels_model == 1: 
                mixture = mixture.view(2,1,-1) # [2,1,nb_time_points]
            
            # add padding to the mixture to have a complete window for the last frame
            if modelname in ('open-unmix', 'deep-u-net'):
                mixture, padding = utils.pad_for_stft(mixture,args['args']['nhop'])
            
            Vj = unmix_target(mixture,phoneme,offset).cpu().detach().numpy()
            
            # Revert to channel dimension if mono model
            if nb_channels_model == 1: 
                if modelname in ('open-unmix', 'deep-u-net'):
                    Vj = Vj.reshape(Vj.shape[0],1,2,-1)
                    # out [nb_frames, 1, 2, nb_bins]
                elif modelname in ('convtasnet'):
                    Vj = Vj.transpose(2,1,0,3)
                    # out [1, C, nb_channels, nb_timesteps]
            
            source_names += [target]
            
            if modelname in ('open-unmix', 'deep-u-net') and softmask:
                # only exponentiate the model if we use softmask
                Vj = Vj**alpha
    
            
            if modelname in ('open-unmix','deep-u-net'):
                V.append(Vj[:, 0, ...])  # remove sample dim
            if modelname in ('convtasnet'):
                V.append(Vj[0,0].T/np.max(np.abs(Vj[0,0]))) # voice
                if args['args']['joint']:
                    source_names.append('accompaniment')
                    V.append(Vj[0,1].T/np.max(np.abs(Vj[0,1]))) # accompaniment
        
        
        estimates = {}
        
        if modelname in ('open-unmix','deep-u-net'):
            V = np.transpose(np.array(V), (1, 3, 2, 0))
            #V mask of shape (nb_frames, nb_bins, 2,nb_targets), real values
            
            X = unmix_target.stft(mixture).detach().cpu().numpy()
    
            # convert to complex numpy type
            X = X[..., 0] + X[..., 1]*1j
            X = X[0].transpose(2, 1, 0)
            # X shape (nb_frames, nb_bins, 2). Complex numbers
        
        if modelname == 'open-unmix': # with wiener filtering
            if residual_model or len(targets) == 1:
                V = norbert.residual_model(V, X, alpha if softmask else 1)
                source_names += (['residual'] if len(targets) > 1
                                else ['accompaniment'])
            
            Y = norbert.wiener(V, X.astype(np.complex128), niter,
                            use_softmask=softmask)
            # Y shape (nb_frames, nb_bins, 2, nb_targets), complex
            
            for j, name in enumerate(source_names):
                audio_hat = istft(
                    Y[..., j].T,
                    n_fft=unmix_target.stft.n_fft,
                    n_hopsize=unmix_target.stft.n_hop
                ) # shape [nb_channels, nb_time_frames]
                
                if padding > 0: # remove padding added for complete stft
                    audio_hat = audio_hat[...,:-padding] 
                
                estimates[name] = audio_hat.T
        
        if modelname == 'deep-u-net': # without wiener filtering
            phase_audio = np.angle(X)[...,np.newaxis]
            Y = phase_audio * V
            
            for j, name in enumerate(source_names):
                audio_hat = istft(
                    Y[..., j].T,
                    n_fft=unmix_target.stft.n_fft,
                    n_hopsize=unmix_target.stft.n_hop
                )
                estimates[name] = audio_hat.T
        
        if modelname == 'convtasnet':
            for j, name in enumerate(source_names):
                estimates[name] = V[j]
        
    return estimates


def inference_args(parser, remaining_args):
    inf_parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parser],
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    inf_parser.add_argument(
        '--softmask',
        dest='softmask',
        action='store_true',
        help=('if enabled, will initialize separation with softmask.'
              'otherwise, will use mixture phase with spectrogram')
    )

    inf_parser.add_argument(
        '--niter',
        type=int,
        default=1,
        help='number of iterations for refining results.'
    )

    inf_parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='exponent in case of softmask separation'
    )

    inf_parser.add_argument(
        '--samplerate',
        type=int,
        default=44100,
        help='model samplerate'
    )

    inf_parser.add_argument(
        '--residual-model',
        action='store_true',
        help='create a model for the residual'
    )
    return inf_parser.parse_args()


def test_main(
    input_files=None, samplerate=44100, niter=1, alpha=1.0,
    softmask=False, residual_model=False, model='umxhq',
    targets=('vocals', 'drums', 'bass', 'other'),
    outdir=None, start=0.0, duration=-1.0, no_cuda=False):

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    for input_file in input_files:
        # handling an input audio path
        info = sf.info(input_file)
        start = int(start * info.samplerate)
        # check if dur is none
        if duration > 0:
            # stop in soundfile is calc in samples, not seconds
            stop = start + int(duration * info.samplerate)
        else:
            # set to None for reading complete file
            stop = None

        audio, rate = sf.read(
            input_file,
            always_2d=True,
            start=start,
            stop=stop
        ) # audio is a numpy array with size (nb_timesteps, nb_channels)

        if audio.shape[1] > 2:
            warnings.warn(
                'Channel count > 2! '
                'Only the first two channels will be processed!')
            audio = audio[:, :2]

        if rate != samplerate:
            warnings.warn('Sample rate indicated different than real!')
            # resample to model samplerate if needed
            audio = resampy.resample(audio, rate, samplerate, axis=0)

        if audio.shape[1] == 1:
            # if we have mono, let's duplicate it
            audio = np.repeat(audio, 2, axis=1)

        estimates = separate(
            audio,
            targets=targets,
            model_name=model,
            niter=niter,
            alpha=alpha,
            softmask=softmask,
            residual_model=residual_model,
            device=device
        ) # is a dictionary

        # Set in which folder the results should be put
        if not outdir:
            model_path = Path(model)
            if not model_path.exists():
                output_path = Path(Path(input_file).stem + '_' + model)
            else:
                output_path = Path(
                    Path(input_file).stem + '_' + model_path.stem
                )
        else:
            if len(input_files) > 1:
                output_path = Path(outdir) / Path(input_file).stem
            else:
                output_path = Path(outdir)

        output_path.mkdir(exist_ok=True, parents=True)
                
        for target, estimate in estimates.items():
            sf.write(
                str(output_path / Path(target).with_suffix('.wav')),
                estimate,
                samplerate
            )


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(
        description='OSU Inference',
        add_help=False
    )

    parser.add_argument(
        'input',
        type=str,
        nargs='+',
        help='List of paths to wav/flac files.'
    )

    parser.add_argument(
        '--targets',
        nargs='+',
        default=['vocals', 'drums', 'bass', 'other'],
        type=str,
        help='provide targets to be processed. \
              If none, all available targets will be computed'
    )

    parser.add_argument(
        '--outdir',
        type=str,
        help='Results path where audio evaluation results are stored'
    )

    parser.add_argument(
        '--start',
        type=float,
        default=0.0,
        help='Audio chunk start in seconds'
    )

    parser.add_argument(
        '--duration',
        type=float,
        default=-1.0,
        help='Audio chunk duration in seconds, negative values load full track'
    )

    parser.add_argument(
        '--model',
        default='umxhq',
        type=str,
        help='path to mode base directory of pretrained models'
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA inference'
    )
    
    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)
    
    test_main(
        input_files=args.input, samplerate=args.samplerate,
        alpha=args.alpha, softmask=args.softmask, niter=args.niter,
        residual_model=args.residual_model, model=args.model,
        targets=args.targets, outdir=args.outdir, start=args.start,
        duration=args.duration, no_cuda=args.no_cuda)