import torch
import torch.nn as nn

class Print(nn.Module):
    """
    Layer that prints the size of its input.
    Used to debug nn.Sequential
    """

    def __init__(self, prefix=''):
        super().__init__()
        self.prefix = prefix

    def forward(self, x):
        if self.prefix:
            print(str(self.prefix) + ':', x.size())
        else:
            print(x.size())

        return x

class View(nn.Module):
    """
    Reshape a tensor
    """

    def __init__(self, out_shape):
        super().__init__()
        self.out_shape = out_shape

    def forward(self, input):
        return input.view(self.out_shape)

class Flatten(nn.Module):
    """
    Flatten layer, to flatten convolutional layer output
    """

    def forward(self, input):
        return input.view(input.size(0), -1)

def model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AudioCritic(nn.Module):
    """
    Base class for DeepDrummer audio critic models
    """

    def __init__(
        self,
        n_conv,
        n_mlp,
        p_dropout
    ):
        super().__init__()

        # Number of convolutional features
        self.n_conv = n_conv

        # Size of MLP layers in the tail part of the model
        self.n_mlp = n_mlp

        # Dropout probability
        self.p_dropout = p_dropout

        # Head part of the model
        self.head = self.make_head()

        # Run a dummy forward pass through the head so we
        # can know the size of the middle embedding
        self.eval()
        emb = self.head(torch.zeros((1, 1, 44100*2)))
        emb = emb.view(emb.size(0), -1)
        emb_size = emb.size(-1)
        self.train()

        # Tail/MLP part of the model
        self.tail = self.make_tail(in_size=emb_size)
        self.tail = self.tail.to(self.device)

    @property
    def num_params(self):
        """
        Number of parameters in the model
        """

        return model_size(self)

    @property
    def device(self):
        """
        Get the device the model is stored on
        """

        params = self.parameters()
        return next(params).device

    def make_head(self):
        """
        Create the head part of the model
        """

        raise NotImplementedError

    def make_tail(self, in_size):
        """
        Create the tail part of the model
        """

        return nn.Sequential(
            nn.Linear(in_size, self.n_mlp),
            nn.LeakyReLU(),

            nn.Linear(self.n_mlp, 2),
            nn.LeakyReLU(),

            #Print(),

            nn.LogSoftmax(dim=1),
        )

    def forward(self, chunk):
        """
        Default forward pass
        """

        emb = self.head(chunk)

        # Flatten the embedding
        emb = emb.view(emb.size(0), -1)

        probs = self.tail(emb)

        return probs

    def eval_audio(model, audio):
        """
        Evaluate an audio clip.
        Produce a pseudo-probability that this audio is "good"
        """

        with torch.no_grad():
            # Only evaluate the first bar (two seconds at 120bpm)
            audio = audio[:2*44100]

            audio = torch.from_numpy(audio).to(model.device)
            audio = audio.unsqueeze(dim=0).unsqueeze(dim=0)

            model.eval()
            log_probs = model(audio)
            model.train()

            probs = log_probs.exp().detach().cpu().numpy()
            p_good = probs[0, 1]

        return p_good

class ConvCritic(AudioCritic):
    """
    Convolutional audio critic
    """

    def make_head(self):
        return nn.Sequential(
            nn.Conv1d(1, self.n_conv, kernel_size=8, stride=1),
            nn.MaxPool1d(4, stride=4),
            nn.LeakyReLU(),
            nn.Dropout(self.p_dropout),
            nn.BatchNorm1d(self.n_conv),

            nn.Conv1d(self.n_conv, self.n_conv, kernel_size=8, stride=1),
            nn.MaxPool1d(4, stride=4),
            nn.LeakyReLU(),
            nn.Dropout(self.p_dropout),
            nn.BatchNorm1d(self.n_conv),

            nn.Conv1d(self.n_conv, self.n_conv, kernel_size=8, stride=1),
            nn.MaxPool1d(4, stride=4),
            nn.LeakyReLU(),
            nn.Dropout(self.p_dropout),
            nn.BatchNorm1d(self.n_conv),

            nn.Conv1d(self.n_conv, self.n_conv, kernel_size=8, stride=1),
            nn.MaxPool1d(4, stride=4),
            nn.LeakyReLU(),
            nn.Dropout(self.p_dropout),
            nn.BatchNorm1d(self.n_conv),

            nn.Conv1d(self.n_conv, self.n_conv, kernel_size=8, stride=1),
            nn.MaxPool1d(4, stride=4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.n_conv),

            nn.Conv1d(self.n_conv, self.n_conv, kernel_size=8, stride=1),
            nn.MaxPool1d(4, stride=4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.n_conv),
        )

class SpectreCritic(AudioCritic):
    """
    Critic using a spectrogram/FFT transform
    """

    def __init__(
        self,
        n_conv,
        n_mlp,
        p_dropout,
        n_fft=200
    ):
        self.n_fft = n_fft

        super().__init__(
            n_conv,
            n_mlp,
            p_dropout
        )

    def make_head(self):
        import torchaudio

        return nn.Sequential(
            #Print('pre-conv'),

            torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft,
                win_length=None,
                hop_length=None,
                pad=0,
                power=1,
                normalized=False
            ),

            nn.Conv2d(1, self.n_conv, kernel_size=(4, 8), stride=(2, 4)),
            nn.LeakyReLU(),
            nn.Dropout(self.p_dropout),
            nn.BatchNorm2d(self.n_conv),

            nn.Conv2d(self.n_conv, self.n_conv, kernel_size=(4, 8), stride=(2, 4)),
            nn.LeakyReLU(),
            nn.Dropout(self.p_dropout),
            nn.BatchNorm2d(self.n_conv),

            nn.Conv2d(self.n_conv, self.n_conv, kernel_size=(4, 8), stride=(2, 4)),
            nn.LeakyReLU(),
            nn.Dropout(self.p_dropout),
            nn.BatchNorm2d(self.n_conv),

            nn.Conv2d(self.n_conv, self.n_conv, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Dropout(self.p_dropout),
            nn.BatchNorm2d(self.n_conv),

            nn.Conv2d(self.n_conv, 8, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8),
        )

class MFCCCritic(AudioCritic):
    def __init__(
        self,
        n_conv,
        n_mlp,
        p_dropout,
        n_mfccs=60,
        log_mels=False
    ):
        self.n_mfccs = n_mfccs
        self.log_mels = log_mels

        super().__init__(
            n_conv,
            n_mlp,
            p_dropout
        )

    def make_head(self):
        import torchaudio

        return nn.Sequential(
            torchaudio.transforms.MFCC(
                n_mfcc=self.n_mfccs,
                dct_type=2,
                norm='ortho',
                log_mels=self.log_mels
            ),

            nn.Conv2d(1, self.n_conv, kernel_size=(4, 8), stride=(2, 4)),
            nn.LeakyReLU(),
            nn.Dropout(self.p_dropout),
            nn.BatchNorm2d(self.n_conv),

            nn.Conv2d(self.n_conv, self.n_conv, kernel_size=(4, 8), stride=(2, 4)),
            nn.LeakyReLU(),
            nn.Dropout(self.p_dropout),
            nn.BatchNorm2d(self.n_conv),

            nn.Conv2d(self.n_conv, self.n_conv, kernel_size=(4, 8), stride=(2, 4)),
            nn.LeakyReLU(),
            nn.Dropout(self.p_dropout),
            nn.BatchNorm2d(self.n_conv),

            nn.Conv2d(self.n_conv, 8, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8),
        )
