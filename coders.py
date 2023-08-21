import torch
import msgpack
import warnings

from torch import nn
from model import Encoder as AEEncoder, Decoder as AEDecoder
from compressai.entropy_models import EntropyBottleneck as AEBottleneck


class _AEModule:
    def __init__(
        self,
        weights_path: str,
        device: torch.device,
        verbose: bool,
        type: str,
    ):
        self.verbose = verbose
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        state_dict = torch.load(weights_path, map_location=self._device)
        self._N, self._M = state_dict["N"], state_dict["M"]

        if self.verbose:
            print(f"{self._N} wide", f"{self._M} hidden layers")

        reduce = lambda key: ".".join(key.split(".")[1:])
        self._subdicts = {key: {} for key in ["_bottleneck", type]}
        for k, v in state_dict["state_dict"].items():
            k = reduce(k)
            for key in self._subdicts.keys():
                if k.startswith(key):
                    self._subdicts[key][reduce(k)] = v

        self._bottleneck = AEBottleneck(channels=self._N)
        self._bottleneck.to(self._device)
        self._bottleneck.load_state_dict(self._subdicts["_bottleneck"])
        self._bottleneck.eval()
        self._bottleneck.update()


class Encoder(_AEModule):
    def __init__(
        self,
        weights_path: str,
        device: torch.device = None,
        verbose: bool = True,
    ):
        super().__init__(weights_path, device, verbose, "_encoder")

        self._encoder = AEEncoder(self._N, self._M)
        self._encoder.to(self._device)
        self._encoder.load_state_dict(self._subdicts["_encoder"])
        self._encoder.eval()

        del self._subdicts

    def __call__(self, x: torch.Tensor) -> bytes:
        return self.encode(x)

    def encode(self, x: torch.Tensor) -> bytes:
        if len(x.shape) != 4:
            raise ValueError("Expected four tensor dimensions")

        _, C, H, W = x.shape

        if C != 3:
            raise ValueError("Expected three-channel image")

        if x.max() > 1.0 and x.min() < 0:
            warnings.warn("Tensor is not normalized to [0,1]")

        reduction = 2 ** (2 + self._M)  # input and output layers + hidden layers

        if H / reduction % 1 != 0 or W / reduction % 1 != 0:
            warnings.warn(
                f"Dividing each input image dimension ({H}x{W}) by 2**{2 + self._M} results in truncation. Decoded dimensions will be different."
            )

        with torch.no_grad():
            x = x.to(self._device)
            y = self._encoder(x)
            y_ = self._bottleneck.compress(y)

        return msgpack.packb({"latent": y_, "shape": y.shape[2:]}, use_bin_type=True)


class Decoder(_AEModule):
    def __init__(
        self,
        weights_path: str,
        device: torch.device = None,
        verbose: bool = True,
    ):
        super().__init__(weights_path, device, verbose, "_decoder")

        self._decoder = AEDecoder(self._N, self._M)
        self._decoder.to(self._device)
        self._decoder.load_state_dict(self._subdicts["_decoder"])
        self._decoder.eval()

        del self._subdicts

    def __call__(self, packet: bytes) -> torch.Tensor:
        return self.decode(packet)

    def decode(self, packet: bytes) -> torch.Tensor:
        msg = msgpack.unpackb(packet, use_list=False)

        if not ("latent" in msg and "shape" in msg):
            raise ValueError("Malformed packet")

        latent, shape = msg["latent"], msg["shape"]

        with torch.no_grad():
            y_ = self._bottleneck.decompress(latent, shape)
            x_ = self._decoder(y_)

        if self.verbose:
            N, _, H, W = x_.shape
            print(
                f"{N}x{H}x{W}",
                sum([len(latent[i]) for i in range(len(latent))]) * 8 / N / H / W,
                "bpp",
            )

        return x_
