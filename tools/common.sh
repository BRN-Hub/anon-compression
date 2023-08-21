# https://stackoverflow.com/a/39087286
gdrive_download() {
        FILEID=$1
        FILENAME=$2
        wget -q --show-progress --load-cookies /tmp/__cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/__cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $FILENAME && rm -rf /tmp/__cookies.txt
}

ensure_gum() {
	command -v gum &> /dev/null || { echo >&2 'gum not in $PATH | github.com/charmbracelet/gum'; exit 1; }
}

ensure_root() {
	[[ $(id -u) == 0 ]] || { echo >&2 'missing root privileges'; exit 1; }
}

ensure_integer() {
	[[ "$1" =~ ^[0-9]+$ ]] || { echo >&2 "$1 is not an integer"; exit 1; }
}

