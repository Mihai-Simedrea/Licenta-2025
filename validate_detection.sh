#!/usr/bin/env bash

if [ $# -ne 2 ]; then
  echo "Usage: $0 [resnet|densenet] /path/to/checkpoint.pth"
  exit 1
fi

BACKBONE=$1
CHECKPOINT=$2

python detection_val.py "$BACKBONE" "$CHECKPOINT"
