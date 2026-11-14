#!/bin/sh
# Build the three final files (silent) from the rendered picture.
# Usage: sh tools/make_final.sh [out_dir]     run from the project folder, after render_full.py; out_dir defaults to final
set -e
PIC=out/final_picture_1440.mp4; MASTER=out/final_picture_master.mp4; D=${1:-final}; N=night-bus-wenjing
mkdir -p "$D"
ffmpeg -v error -y -i "$PIC" -map 0:v -c:v copy -an -movflags +faststart "$D/${N}_silent_10s_1440.mp4"
ffmpeg -v error -y -stream_loop 2 -i "$PIC" -map 0:v -c:v copy -an -t 30 -movflags +faststart "$D/${N}_silent_30s_1440.mp4"
ffmpeg -v error -y -stream_loop 2 -i "$MASTER" -map 0:v -c:v copy -an -t 30 -movflags +faststart "$D/${N}_silent_30s_2048.mp4"
ls -la "$D"
