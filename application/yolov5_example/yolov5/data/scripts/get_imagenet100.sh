#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Download ILSVRC2012 ImageNet dataset https://image-net.org
# Example usage: bash data/scripts/get_imagenet.sh
# parent
# ├── yolov5
# └── datasets
#     └── imagenet  ← downloads here

# Arguments (optional) Usage: bash data/scripts/get_imagenet.sh --train --val
if [ "$#" -gt 0 ]; then
  for opt in "$@"; do
    case "${opt}" in
    --train) train=true ;;
    --val) val=true ;;
    esac
  done
else
  train=true
  val=true
fi

# Make dir
d='../datasets/imagenet100' # unzip directory
mkdir -p $d && cd $d

# Download/unzip train
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/imagenet100.zip
unzip imagenet100.zip && rm imagenet100.zip
