#!/bin/bash

set -e

source tools/common.sh

ensure_gum

echo 'Choose architecture configuration'
CONFIG=$(gum choose '(256,2)' '(128,2)' '(256,1)' '(256,2) - compat' '(128,2) - compat')

mkdir -p weights

case "$CONFIG" in
	'(256,2)')
		echo 'Width 256, Depth 2'
#		gdrive_download FIXME weights/256-2.pt
		;;
	'(128,2)')
		echo 'Width 128, Depth 2'
#		gdrive_download FIXME weights/128-2.pt
		;;
	'(256,1)')
		echo 'Width 256, Depth 1'
#		gdrive_download FIXME weights/256-1.pt
		;;
	'(256,2) - compat')
		echo 'Width 256, Depth 1, jetson nano compatibility'
#		gdrive_download FIXME weights/256-2-nanocompat.pt
		;;
	'(128,2) - compat')
		echo 'Width 128, Depth 2, jetson nano compatibility'
#		gdrive_download FIXME weights/128-2-nanocompat.pt
		;;
esac

