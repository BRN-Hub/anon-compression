import os
import torch
import torchvision
import pandas

import lightning.pytorch as pl

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.io import read_image, ImageReadMode
from glob import glob
from pybboxes import convert_bbox


def create_ann_dict(
    tuple,
    annotation_format,
    with_confidence,
    with_class,
    img_h,
    img_w,
    f_x,
    f_y,
):
    x, y, w, h = tuple[-4::]
    x = max(0, x)
    y = max(0, y)
    if x + w > img_w:
        w = img_w - x
    if y + h > img_h:
        h = img_h - y

    if w <= 0 or h <= 0:
        return None

    if annotation_format != "voc":
        x, y, w, h = convert_bbox(
            (x, y, w, h),
            from_type=annotation_format,
            to_type="voc",
            image_size=(img_w, img_h),
        )

    dict = {
        "xyxy": (
            int(x * f_x),
            int(y * f_y),
            int(w * f_x),
            int(h * f_y),
        ),
    }

    if with_class:
        dict["class"] = tuple[0]
    if with_confidence:
        dict["conf"] = tuple[1]

    return dict


class CrowdhumanAnnotatedDataset(Dataset):
    def __init__(
        self,
        image_folder_path,
        yolo_annotations_path=None,
        facenet_annotations_path=None,
        annotation_format="yolo",
        with_confidence=False,
        transform=None,
    ):
        self.images = glob(image_folder_path + "*.jpg")
        self.yolo_anns = yolo_annotations_path
        self.face_anns = facenet_annotations_path
        self.transform = transform
        self.annotation_format = annotation_format
        self.with_confidence = with_confidence

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        assert not torch.is_tensor(idx), "Didn't expect indexing tensor!"

        image_path = self.images[idx]
        ann_filename = os.path.splitext(image_path)[0].split("/")[-1] + ".txt"

        image = read_image(image_path, ImageReadMode.RGB) / 255.0

        _, H, W = image.shape

        if self.transform:
            image = self.transform(image)

        _, H_, W_ = image.shape

        scale_x = W_ / W
        scale_y = H_ / H

        sample = {
            "image": image,
            "yolo": [],
            "facenet": [],
        }

        if self.yolo_anns is not None:
            try:
                frame = pandas.read_csv(
                    self.yolo_anns + ann_filename,
                    delim_whitespace=True,
                    header=None,
                )
            except pandas.errors.EmptyDataError:
                frame = pandas.DataFrame()

            sample["yolo"] = [
                d
                for d in [
                    create_ann_dict(
                        tuple,
                        self.annotation_format,
                        self.with_confidence,
                        with_class=True,
                        img_h=H,
                        img_w=W,
                        f_x=scale_x,
                        f_y=scale_y,
                    )
                    for tuple in frame.itertuples(index=False, name=None)
                ]
                if d is not None
            ]

        if self.face_anns is not None:
            try:
                frame = pandas.read_csv(
                    self.face_anns + ann_filename,
                    delim_whitespace=True,
                    header=None,
                )
            except pandas.errors.EmptyDataError:
                frame = pandas.DataFrame()

            sample["facenet"] = [
                d
                for d in [
                    create_ann_dict(
                        tuple,
                        self.annotation_format,
                        self.with_confidence,
                        with_class=False,
                        img_h=H,
                        img_w=W,
                        f_x=scale_x,
                        f_y=scale_y,
                    )
                    for tuple in frame.itertuples(index=False, name=None)
                ]
                if d is not None
            ]

        return sample


class CrowdhumanAnnotatedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_folder_path,
        yolo_annotations_path=None,
        facenet_annotations_path=None,
        annoatation_format="yolo",
        with_confidence=False,
        transform=None,
        batch_size=32,
        validation_split=0.15,
        num_gpus=1,
    ):
        super().__init__()
        self.image_folder_path = image_folder_path
        self.yolo_annotations_path = yolo_annotations_path
        self.facenet_annotations_path = facenet_annotations_path
        self.with_confidence = with_confidence
        self.annotation_format = annoatation_format

        self.validation_split = validation_split
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.transform = transform

    @staticmethod
    def collate_fn(batch):
        batch_tensor = torch.stack([item["image"] for item in batch])
        yolo_annotations = [item["yolo"] for item in batch]
        facenet_annotations = [item["facenet"] for item in batch]
        return batch_tensor, yolo_annotations, facenet_annotations

    def setup(self, stage: str):
        self.dataset_train = CrowdhumanAnnotatedDataset(
            self.image_folder_path + "/train/",
            yolo_annotations_path=(self.yolo_annotations_path + "/")
            if self.yolo_annotations_path is not None
            else None,
            facenet_annotations_path=(self.facenet_annotations_path + "/")
            if self.facenet_annotations_path is not None
            else None,
            transform=self.transform,
            annotation_format=self.annotation_format,
            with_confidence=self.with_confidence,
        )
        self.dataset_val = self.dataset_train = CrowdhumanAnnotatedDataset(
            self.image_folder_path + "/val/",
            yolo_annotations_path=(self.yolo_annotations_path + "/")
            if self.yolo_annotations_path is not None
            else None,
            facenet_annotations_path=(self.facenet_annotations_path + "/")
            if self.facenet_annotations_path is not None
            else None,
            transform=self.transform,
            annotation_format=self.annotation_format,
            with_confidence=self.with_confidence,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_gpus * 8,
            collate_fn=CrowdhumanAnnotatedDataModule.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            self.batch_size,
            pin_memory=True,
            num_workers=self.num_gpus * 8,
            collate_fn=CrowdhumanAnnotatedDataModule.collate_fn,
        )
