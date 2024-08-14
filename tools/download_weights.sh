#!/bin/bash

set -e

source tools/common.sh

ensure_gum

echo 'Choose architecture configuration'
CONFIG=$(gum choose '(256,2)' '(128,2)' '(256,1)' '(256,2) - compat' '(128,2) - compat')

mkdir -p weights
cd weights

BASEURL="https://robotik.informatik.uni-wuerzburg.de/telematics/exchange/anon-compression"

case "$CONFIG" in
	'(256,2)')
		echo 'Width 256, Depth 2'
		http_download "$BASEURL/256-2.pt"
		;;
	'(128,2)')
		echo 'Width 128, Depth 2'
		http_download "$BASEURL/128-2.pt"
		;;
	'(256,1)')
		echo 'Width 256, Depth 1'
		http_download "$BASEURL/256-1.pt"
		;;
	'(256,2) - compat')
		echo 'Width 256, Depth 1, jetson nano compatibility'
		http_download "$BASEURL/256-2-nanocompat.pt"
		;;
	'(128,2) - compat')
		echo 'Width 128, Depth 2, jetson nano compatibility'
		http_download "$BASEURL/128-2-nanocompat.pt"
		;;
esac

if ! [ -f 'sha256sums.txt' ]; then
	http_download "$BASEURL/sha256sums.txt"
fi

sha256sum --ignore-missing --check sha256sums.txt
