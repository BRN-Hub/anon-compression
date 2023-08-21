#!/usr/bin/env python3

import os, shutil
import torch

import torchvision.transforms.functional as F

from torch.nn.functional import mse_loss
from torchvision import transforms
from lightning import pytorch as pl
from ultralytics import YOLO
from uuid import uuid4

from metrics.BoundingBox import BoundingBox as BBox, BBType, BBFormat
from metrics.BoundingBoxes import BoundingBoxes as BBoxes
from metrics.utils import CoordinatesType as BBCords
from metrics.Evaluator import Evaluator as BBMetrics
from pybboxes import convert_bbox

from model import AutoEncoder
from datamodule import CrowdhumanAnnotatedDataModule
from region_loss import region_loss


def main():
    batch_size=14
    model = AutoEncoder(
        N=128,
        M=2,
    )
    train_module = LightningWrapper(
        model,
        batch_size=batch_size,
        bg_factor=1.0,
        vbox_factor=1.0,
        hbox_factor=0.6,
        learning_rate=1e-4,
        rate_factor=0.04,
        val_iou_threshold=0.5,
    )

    n_gpus = torch.cuda.device_count()

    trainer = pl.Trainer(
        limit_train_batches=100,
        max_epochs=300,
        accelerator="auto",
        devices=n_gpus,
    )

    dataset = CrowdhumanAnnotatedDataModule(
        "data/crowdhuman",
        yolo_annotations_path="data/crowdhuman/annotations/vbox",
        facenet_annotations_path="data/crowdhuman/annotations/hbox",
        transform=transforms.Resize((512, 512), antialias=True),
        num_gpus=n_gpus,
        batch_size=batch_size,
        with_confidence=False,
        annoatation_format="coco",
    )

    trainer.fit(train_module, dataset)

    out_path = f"./out/{uuid4()}"
    os.makedirs(out_path)
    print("=> Saving to", out_path)

    autoencoder_path = f"{out_path}/autoencoder.pth"
    state_dict = {
        "state_dict": model.state_dict(),
        "N": model.N,
        "M": model.M,
    }

    torch.save(state_dict, autoencoder_path)


class LightningWrapper(pl.LightningModule):
    def __init__(
        self,
        autoencoder,
        bg_factor,
        vbox_factor,
        hbox_factor,
        learning_rate,
        rate_factor,
        batch_size,
        val_iou_threshold,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.yolo = YOLO("yolov8x.pt")
        self.bg_factor = bg_factor
        self.vbox_factor = vbox_factor
        self.hbox_factor = hbox_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rate_factor = rate_factor
        self.iou_threshold = val_iou_threshold

        self.save_hyperparameters()

        # manual opt
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        loss, _, bottleneck_loss = self._step(batch, batch_idx, stage="train")

        self.log("train_loss", loss, batch_size=self.batch_size)

        opt, aux_opt = self.optimizers()

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        aux_opt.zero_grad()
        self.manual_backward(bottleneck_loss)
        aux_opt.step()

    def validation_step(self, batch, batch_idx):
        input, yolo_gt, _ = batch
        loss, output, _ = self._step(batch, batch_idx, stage="val")
        ap, _, _ = self._invoke_yolo(yolo_gt, output)

        self.log("val_loss", loss, batch_size=self.batch_size)
        self.log("yolo_human_AP", ap, batch_size=self.batch_size)

    def _step(self, batch, batch_idx, stage):
        input, yolo, face = batch

        N, _, H, W = input.shape

        output = self.autoencoder.forward(input)

        reconstruction_loss = mse_loss(output, input)
        yolo_loss = region_loss(input, output, yolo, invert=False)
        face_loss = region_loss(input, output, face, invert=True)
        bpp_loss = -torch.log2(self.autoencoder.likelihoods).sum() / (N * H * W)

        self.log(stage + "_reconstruction_loss", reconstruction_loss, batch_size=self.batch_size)
        self.log(stage + "_bpp_loss", bpp_loss, batch_size=self.batch_size)
        self.log(stage + "_face_loss", face_loss, batch_size=self.batch_size)
        self.log(stage + "_yolo_loss", yolo_loss, batch_size=self.batch_size)

        loss = (
            reconstruction_loss * self.bg_factor
            + face_loss * self.hbox_factor
            + yolo_loss * self.vbox_factor
            + bpp_loss * self.rate_factor
        )

        return loss, output, self.autoencoder._bottleneck.loss()

    def _invoke_yolo(self, ground_truth, compressed):
        results = self.yolo(compressed, classes=0, verbose=False)  # human
        boxes = BBoxes()
        for batch_idx, result in enumerate(results):
            if result.boxes.xyxy.shape[0] == 0:
                continue
            for (x_tl, y_tl, x_br, y_br), conf in zip(
                result.boxes.xyxy, result.boxes.conf
            ):
                boxes.addBoundingBox(
                    BBox(
                        imageName=str(batch_idx),
                        classId="human",
                        x=x_tl,
                        y=y_tl,
                        w=x_br,
                        h=y_br,
                        typeCoordinates=BBCords.Absolute,
                        bbType=BBType.Detected,
                        format=BBFormat.XYX2Y2,
                        classConfidence=conf,
                    )
                )
        for batch_idx, item in enumerate(ground_truth):
            if len(item) == 0:
                continue
            for detection in item:
                x_tl, y_tl, x_br, y_br = detection["xyxy"]
                boxes.addBoundingBox(
                    BBox(
                        imageName=str(batch_idx),
                        classId="human",
                        x=x_tl,
                        y=y_tl,
                        w=x_br,
                        h=y_br,
                        typeCoordinates=BBCords.Absolute,
                        bbType=BBType.GroundTruth,
                        format=BBFormat.XYX2Y2,
                    )
                )
        metrics = BBMetrics().GetPascalVOCMetrics(
            boxes, IOUThreshold=self.iou_threshold
        )[
            0
        ]  # only one class
        return metrics["AP"], metrics["precision"], metrics["recall"]

    def configure_optimizers(self):
        parameters = set(
            p
            for n, p in self.autoencoder.named_parameters()
            if not n.endswith(".quantiles")
        )
        aux_parameters = set(
            p
            for n, p in self.autoencoder.named_parameters()
            if n.endswith(".quantiles")
        )
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        aux_optimizer = torch.optim.Adam(aux_parameters, lr=1e-3)
        return optimizer, aux_optimizer


if __name__ == "__main__":
    main()
