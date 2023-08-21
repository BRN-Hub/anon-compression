import torch

from torch import nn
from compressai.layers import GDN1
from compressai.models.utils import conv, deconv
from compressai.entropy_models import EntropyBottleneck


class Encoder(nn.Module):
    def __init__(self, N, M):
        super().__init__()

        layers = [
            conv(3, N),
            GDN1(N),
        ]
        for _ in range(M):
            layers.extend(
                [
                    conv(N, N),
                    GDN1(N),
                ]
            )

        layers.append(conv(N, N))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, N, M):
        super().__init__()

        layers = [
            deconv(N, N),
            GDN1(N, inverse=True),
        ]

        for _ in range(M):
            layers.extend(
                [
                    deconv(N, N),
                    GDN1(N, inverse=True),
                ]
            )

        layers.extend(
            [
                deconv(N, 3),
                nn.Sigmoid(),
            ]
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class AutoEncoder(nn.Module):
    def __init__(self, N=256, M=2):
        super().__init__()

        self.N = N
        self.M = M
        self._encoder = Encoder(N, M)
        self._decoder = Decoder(N, M)
        self._bottleneck = EntropyBottleneck(channels=N)

    def forward(self, x):
        y = self._encoder(x)

        y_, self.likelihoods = self._bottleneck(y, training=self.training)

        return self._decoder(y_)

    def encoder(self):
        return self._encoder

    def decoder(self):
        return self._decoder
