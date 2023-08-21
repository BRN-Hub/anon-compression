#!/bin/bash

set -e

source tools/common.sh

ensure_gum

unzip_move() {
	export OUTDIR=$1
	export ZIPFILES="${@:2}"
	mkdir -p $OUTDIR
	gum spin --show-output --spinner jump --title "Extracting $ZIPFILES to $OUTDIR ..." -- bash -c 'for Z in $ZIPFILES; do unzip -j $Z -d $OUTDIR; done'
	rm $ZIPFILES
}

gdrive_download 18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO val.zip
unzip_move data/crowdhuman/val val.zip

gdrive_download 10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL annotation_val.odgt
ODGT_PATHS="annotation_val.odgt"

if gum confirm "Prepare training data?"; then
	gdrive_download 134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y train01.zip
	gdrive_download 17evzPh7gc1JBNvnW1ENXLy5Kr4Q_Nnla train02.zip
	gdrive_download 1tdp0UCgxrqy1B6p8LkR-Iy0aIJ8l4fJW train03.zip
	unzip_move data/crowdhuman/train train0{1,2,3}.zip

	gdrive_download 1UUTea5mYqvlUObsC1Z8CFldHJAtLtMX3 annotation_train.odgt
	ODGT_PATHS="$ODGT_PATHS annotation_train.odgt"
fi

gum spin --show-output --spinner points --title 'Processing annotations ...' -- python3 tools/odgt_to_yolo.py --odgt_paths $ODGT_PATHS --out_dir data/crowdhuman/annotations
rm $ODGT_PATHS


