from model import Spectrogram, STFT, NoOp
class Model(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        nb_channels=2,
        input_is_spectrogram=False,
        sample_rate=44100,
    ):
        """
        Input:  (batch, channel, sample)
            or  (frame, batch, channels, frequency)
        Output: (frame, batch, channels, frequency)
        """

        super(OpenUnmix, self).__init__()
        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=power, mono=(nb_channels == 1))
        # register sample_rate to check at inference time
        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        if input_is_spectrogram:
            self.transform = NoOp()
        else:
            self.transform = nn.Sequential(self.stft, self.spec)


    def forward(self, mix):
        # transform to spectrogram on the fly
        X = self.transform(mix)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        # transform X to estimate
        # ....

        return X