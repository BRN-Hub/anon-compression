#!/usr/bin/env python3

import sys, os
sys.path.append(os.getcwd())

import torch
import pillow_heif
import pandas
import numpy

from facenet_pytorch import MTCNN
from glob import glob
from io import BytesIO
from common import partition, compress, compress_nn, load_nn, get_annotations
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import to_pil_image, resize
from model import AutoEncoder
from PIL import ImageDraw

from metrics.Evaluator import (
    BoundingBox as BBox,
    BoundingBoxes as BBoxes,
    BBType,
    BBFormat,
    Evaluator as BBMetrics,
    CoordinatesType as BBCoordType,
)

path_images = "data/crowdhuman/val"
path_annotations = "data/crowdhuman/annotations/hbox"

compress_with_nn = True  # switch between codec and autoencoder

codec_ext = "AVIF"  # any pillow codec
quality = 10  # 1 - 100
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

    boxes = []
    for box in results:
        print(len(box))
        boxes.extend(box)

    print("Calculating metrics")

    metrics = BBMetrics().GetPascalVOCMetrics(BBoxes(boxes), iou_threshold)[0]

    if compress_with_nn:
        print(nn_path)
    else:
        print(codec_ext, quality)
    print("IOU", iou_threshold)
    print("AP", metrics["AP"])
    print("TP", metrics["total TP"])
    print("FP", metrics["total FP"])


def process(args):
    global path_annotations, path_images, iou_threshold, nn_path, input_shape, quality, codec_ext, compress_with_nn
    pillow_heif.register_avif_opener()
    chunk, id = args
    device_id = torch.device(f"cuda:{id}")
    mtcnn = MTCNN(device=device_id)
    print(device_id)

    ae = load_nn(device_id, nn_path)

    bounding_boxes = []
    save_example = True

    with torch.no_grad():
        for file in chunk:
            image, _, orig_shape = (
                compress_nn(file, ae, device_id, input_shape)
                if compress_with_nn
                else compress(file, quality, codec_ext, input_shape)
            )

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
                    try:
                        draw.rectangle([(x, y), (x + w, y + h)], outline="green")
                    except:
                        print(x, y, w, h, orig_shape, input_shape)
                        raise

                bounding_boxes.append(
                    BBox(
                        imageName=file,
                        classId="head",
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        typeCoordinates=BBCoordType.Absolute,
                        bbType=BBType.GroundTruth,
                        format=BBFormat.XYWH,
                    )
                )

            try:
                if isinstance(image, torch.Tensor):
                    image = to_pil_image((image.squeeze() * 255.0).byte())
                result = mtcnn.detect(image, landmarks=False)
            except:
                result = [None, None]
            boxes, confs = result

            if boxes is None or confs is None:
                continue  # no detections

            assert len(confs) == len(boxes)

            for (x_tl, y_tl, x_br, y_br), conf in zip(boxes, confs):
                if save_example:
                    draw.rectangle([(x_tl, y_tl), (x_br, y_br)], outline="red")
                bounding_boxes.append(
                    BBox(
                        imageName=file,
                        classId="head",
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
                clone.save(f"/tmp/example_facenet_{id}.png")
                save_example = False

    return bounding_boxes


if __name__ == "__main__":
    main()
