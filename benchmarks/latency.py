#!/usr/bin/env python3

import sys, os
sys.path.append(os.getcwd()) # "relative imports"

import timeit
import torch
import pillow_heif
import numpy
import json

from common import load_nn
from torchvision.transforms.functional import resize
from torchvision.io import read_image, ImageReadMode
from PIL import Image
from io import BytesIO
from glob import glob
from uuid import uuid4


path_images = "data/crowdhuman/val"
ckpt_path = "weights/256-2.pt"

samples_per_image = 10

skip_codecs = False
skip_nn_cpu = True
skip_nn_gpu = False

codecs = ["JPEG", "AVIF"]
qualities = [1, 5, 10]
input_shape = (512, 512)

gpu = torch.device("cuda")
cpu = torch.device("cpu")

if not skip_nn_gpu:
    ae_gpu = load_nn(gpu, ckpt_path)
if not skip_nn_cpu:
    ae_cpu = load_nn(cpu, ckpt_path)

pillow_heif.register_avif_opener()


results = {
    method: {
        "compress": [],
        "decompress": [],
    }
    for method in ["gpu", "cpu"]
}
for codec in codecs:
    results[codec] = {
        quality: {
            "compress": [],
            "decompress": [],
        }
        for quality in qualities
    }

files = glob(path_images + "/*.jpg", recursive=False)

for i, file in enumerate(files):
    print("\n", len(files), i, file, end=" ")
    try:
        tensor_cpu = resize(
            read_image(file, ImageReadMode.RGB).unsqueeze(0) / 255.0,
            input_shape,
            antialias=True,
        )
    except RuntimeError:  # oom
        print("Skipping, could not allocate!")
        continue

    if not skip_nn_cpu:
        y = ae_cpu._encoder(tensor_cpu)
        shape_cpu = y.shape[2:]
        buffer_cpu = ae_cpu._bottleneck.compress(y)

    if not skip_nn_gpu:
        tensor_gpu = tensor_cpu.to(gpu)
        y = ae_gpu._encoder(tensor_gpu)
        shape_gpu = y.shape[2:]
        buffer_gpu = ae_gpu._bottleneck.compress(y)

    image_pil = resize(
        Image.open(file),
        input_shape,
        antialias=True,
    )
    image_pil.load()
    buffer_dict = {}
    for codec in codecs:
        buffer_dict[codec] = {}
        for quality in qualities:
            buffer_dict[codec][quality] = BytesIO()
            image_pil.save(buffer_dict[codec][quality], codec, quality=quality)

    if not skip_nn_gpu:
        results["gpu"]["compress"].extend(
            timeit.repeat(
                stmt=lambda: ae_gpu._bottleneck.compress(ae_gpu._encoder(tensor_gpu)),
                repeat=samples_per_image,
                number=1,
            )
        )
        results["gpu"]["decompress"].extend(
            timeit.repeat(
                stmt=lambda: ae_gpu._decoder(
                    ae_gpu._bottleneck.decompress(buffer_gpu, shape_gpu)
                ),
                repeat=samples_per_image,
                number=1,
            )
        )
        print("gpu", end=" ")

    if not skip_nn_cpu:
        results["cpu"]["compress"].extend(
            timeit.repeat(
                stmt=lambda: ae_cpu._bottleneck.compress(ae_cpu._encoder(tensor_cpu)),
                repeat=samples_per_image,
                number=1,
            )
        )
        results["cpu"]["decompress"].extend(
            timeit.repeat(
                stmt=lambda: ae_cpu._decoder(
                    ae_cpu._bottleneck.decompress(buffer_cpu, shape_cpu)
                ),
                repeat=samples_per_image,
                number=1,
            )
        )
        print("cpu", end=" ")

    if not skip_codecs:
        for codec in codecs:
            for quality in qualities:
                # this buffer may grow very large.
                with BytesIO() as buffer:
                    results[codec][quality]["compress"].extend(
                        timeit.repeat(
                            stmt=lambda: image_pil.save(buffer, codec, quality=quality),
                            repeat=samples_per_image,
                            number=1,
                        )
                    )
                results[codec][quality]["decompress"].extend(
                    timeit.repeat(
                        stmt=lambda: Image.open(buffer_dict[codec][quality]),
                        repeat=samples_per_image,
                        number=1,
                    )
                )
            print(codec, end=" ")

out_file = f"{uuid4()}.json"
print("\n\n==> Saving results to ", out_file)
with open(out_file, "w") as fp:
    json.dump(results, fp)


def pretty_print(d):
    for k, v in d.items():
        print(k, "-> ", end="")
        if isinstance(v, dict):
            pretty_print(v)
        elif len(v) > 1:
            print(
                "Avg:",
                numpy.average(v),
                "Std:",
                numpy.std(v, ddof=1),
                "Min:",
                numpy.min(v),
                "Max:",
                numpy.max(v),
            )


pretty_print(results)
