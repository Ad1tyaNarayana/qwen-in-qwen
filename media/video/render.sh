#!/bin/sh
# Renders every asset from the real captures. Run media/capture/capture.sh first.
set -eu
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
OUT=$ROOT/.cache/media
DOCS=$ROOT/docs/media
mkdir -p "$DOCS"
npx remotion render SocialVertical "$OUT/qwen-in-qwen-vertical.mp4"
npx remotion render SocialLandscape "$OUT/qwen-in-qwen-landscape.mp4"
npx remotion still SocialVertical "$OUT/qwen-in-qwen-vertical-poster.png" --frame=420
npx remotion still SocialLandscape "$OUT/qwen-in-qwen-landscape-poster.png" --frame=420
npx remotion still Preview "$DOCS/social-preview.png"
npx remotion render Hero "$OUT/hero.mp4"
npx remotion render StoryCard "$OUT/story.mp4"
gif() {
	ffmpeg -y -loglevel error -i "$1" -filter_complex \
		"fps=$3,scale=$4:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=$5:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" "$2"
}
gif "$OUT/hero.mp4" "$DOCS/hero.gif" 12 900 128
gif "$OUT/story.mp4" "$DOCS/story.gif" 10 800 32

still() {
	ffmpeg -y -loglevel error -i "$OUT/capture/$1" -vf "crop=$3,scale=1400:-1:flags=lanczos" -frames:v 1 "$DOCS/$2"
}
still orbit/0040.jpg network.png 2240:1280:620:420
still orbit-ndot/0075.jpg accelerator.png 2240:1280:620:420
still l4-app.png model.png 2260:1160:600:250
still life.png life.png 1150:920:650:320
still mnist.png mnist.png 2200:830:640:640
