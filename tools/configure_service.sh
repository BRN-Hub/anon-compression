#!/bin/bash

set -e

source tools/common.sh

ensure_gum

[ -n "$(ls /dev/video*)" ] || { echo >&2 'No video devices available'; exit 1; }
[ -n "$(find weights -name '*.pt' -type f -prune)" ] || { echo >&2 'No network weights available'; exit 1; }

WEIGHTS="weights/$(ls weights/*.pt | xargs -n 1 basename | gum choose --header='What kind of model is supposed to run?' --cursor='👉 ')"

DIMENSION="$(gum input --header='Enter the maximum image side length; One single integer, preferrably a power of two:' --value 512 --prompt='👉 ')"
ensure_integer "$DIMENSION"

CAPTURE="$(ls /dev/video* | gum choose --header='Which video device? (Space to select, enter to finish)' --cursor='👉 ' --no-limit | tr '\n' ' ' | xargs)"

HOST="$(gum input --header='Enter IP to listen on:' --value '0.0.0.0' --prompt='👉 ')"
/usr/bin/env python3 -c "import ipaddress; ipaddress.ip_address('${HOST}')" > /dev/null 2>&1 || { echo >&2 "$HOST is not a valid IP"; exit 1; }

PORT="$(gum input --header='Enter port to listen on:' --value 8888 --prompt='👉 ')"
ensure_integer "$PORT"

cp examples/anon-capture.service.template tools/anon-capture.service

PYTHON="$(pwd)/.venv/bin/python3"
if ! [ -f "$PYTHON" ]; then
	PYTHON="$(command -v python3)"
fi

sed -i "s#WorkingDirectory=FIXME#WorkingDirectory=$(pwd)#g" tools/anon-capture.service
sed -i "s#ExecStart=FIXME#ExecStart=$PYTHON $(pwd)/examples/capture.py $(pwd)/$WEIGHTS --dimension $DIMENSION --capture $CAPTURE --host $HOST --port $PORT#g" tools/anon-capture.service

set +e

gum confirm 'Review unit file?' --affirmative='yep' --negative='back up!' && gum pager < tools/anon-capture.service --border normal --soft-wrap

UNITS_PATH=/etc/systemd/system
gum confirm "Install unit file to $UNITS_PATH?" --affirmative='yep' --negative='back up!' --default='no' || { echo "Configured unit file: tools/anon-capture.service"; exit 0; }
[ -d "$UNITS_PATH" ] || { echo >&2 "$UNITS_PATH does not exist, manual intervention required"; exit 1; }

ensure_root

set -x
mv tools/anon-capture.service "$UNITS_PATH" 
systemctl daemon-reload
systemctl enable --now anon-capture.service
