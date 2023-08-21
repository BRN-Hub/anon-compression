#!/usr/bin/env python3

import sys, os
sys.path.append(os.getcwd())

import torch
import pillow_heif
import pandas
import numpy

from ultralytics import YOLO
from glob import glob
from PIL import Image, ImageDraw
from io import BytesIO
from common import partition, compress, compress_nn, load_nn, get_annotations
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import to_pil_image, resize
from model import AutoEncoder

from metrics.Evaluator import (
    BoundingBox as BBox,
    BoundingBoxes as BBoxes,
    BBType,
    BBFormat,
    Evaluator as BBMetrics,
    CoordinatesType as BBCoordType,
)

path_images = "data/crowdhuman/val"
path_annotations = "data/crowdhuman/annotations/vbox"
classes = 0

compress_with_nn = True  # switch between codec and nn benchmarks

codec_ext = "AVIF"  # any PIL codec
quality = 10  # 1-100
iou_threshold = 0.5

nn_path = "weights/256-2.pt"
input_shape = (512, 512)


def main():
    n_gpus = torch.cuda.device_count()
    torch.multiprocessing.set_start_method("spawn")
    with torch.multiprocessing.Pool(n_gpus) as pool:
        results = pool.map(
            process,
            zip(
                [
                    p
                    for p in partition(
                        glob(path_images + "/*.jpg", recursive=False), n_gpus
                    )
                ],
                [idx for idx in range(n_gpus)],
            ),
        )

    bpps = []
    boxes = []

    for box, bpp in results:
        print(len(box), len(bpp))
        boxes.extend(box)
        bpps.extend(bpp)

    print("Calculating metrics")

    metrics = BBMetrics().GetPascalVOCMetrics(BBoxes(boxes), iou_threshold)[0]

    if compress_with_nn:
        print(nn_path)
    else:
        print(codec_ext, quality)
    print("bpp", numpy.mean(bpps), numpy.std(bpps, ddof=1))
    print("IOU", iou_threshold)
    print("AP", metrics["AP"])
    print("TP", metrics["total TP"])
    print("FP", metrics["total FP"])


def process(args):
    global classes, path_annotations, path_images, iou_threshold, nn_path, input_shape, quality, codec_ext, compress_with_nn
    pillow_heif.register_avif_opener()
    chunk, id = args
    yolo = YOLO("yolov8x.pt")
    device_id = torch.device(f"cuda:{id}")
    print(device_id)
    yolo.to(device_id)

    ae = load_nn(device_id, nn_path)

    bpps = []
    boxes = []
    save_example = True

    with torch.no_grad():
        for file in chunk:
            image, bpp, orig_shape = (
                compress_nn(file, ae, device_id, input_shape)
                if compress_with_nn
                else compress(file, quality, codec_ext, input_shape)
            )

            bpps.append(bpp)

            if save_example:
                clone = (
                    to_pil_image((image.squeeze() * 255).byte())
                    if isinstance(image, torch.Tensor)
                    else image.copy()
                )
                draw = ImageDraw.Draw(clone)

            for x, y, w, h in get_annotations(
                path_annotations, path_images, file, input_shape, orig_shape
            ):
                if save_example:
                    draw.rectangle([(x, y), (x + w, y + h)], outline="green")

                boxes.append(
                    BBox(
                        imageName=file,
                        classId="human",
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        typeCoordinates=BBCoordType.Absolute,
                        bbType=BBType.GroundTruth,
                        format=BBFormat.XYWH,
                    )
                )

            result = yolo(image, classes=classes, verbose=False)[0]

            for (x_tl, y_tl, x_br, y_br), conf in zip(
                result.boxes.xyxy, result.boxes.conf
            ):
                if save_example:
                    draw.rectangle([(x_tl, y_tl), (x_br, y_br)], outline="red")

                boxes.append(
                    BBox(
                        imageName=file,
                        classId="human",
                        x=x_tl,
                        y=y_tl,
                        w=x_br,
                        h=y_br,
                        typeCoordinates=BBCoordType.Absolute,
                        bbType=BBType.Detected,
                        format=BBFormat.XYX2Y2,
                        classConfidence=conf,
                    )
                )
            if save_example:
                clone.save(f"/tmp/example_yolo_{id}.png")
                save_example = False

    return (boxes, bpps)


if __name__ == "__main__":
    main()
