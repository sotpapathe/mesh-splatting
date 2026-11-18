#!/bin/sh
# SPDX-FileCopyrightText: 2026 Sotiris Papatheodorou
# SPDX-License-Identifier: CC0-1.0
set -eu

export DATASET=~/data/datasets/Mip-NeRF-360/room
export DATASET_OPTS='--indoor'
export ITERATIONS=30000 # default

cd "$(dirname "$0")" || exit 1
modeldir=$(basename "$DATASET").model

if [ ! -f "$modeldir/point_cloud/iteration_$ITERATIONS/point_cloud_state_dict.pt" ]
then
	python3 train.py --eval -s "$DATASET" -m "$modeldir" --resolution 8 --iterations "$ITERATIONS" $DATASET_OPTS
fi

if [ ! -f "$modeldir/test/ours_$ITERATIONS/renders/00000.png" ]
then
	python3 render.py --eval -s "$DATASET" -m "$modeldir" --iteration "$ITERATIONS" --skip_train --quiet
fi

if [ ! -f "$modeldir/results.json" ]
then
	python3 metrics.py -m "$modeldir"
fi

if [ ! -f "$modeldir/mesh.ply" ]
then
	python3 create_ply.py --out "$modeldir/mesh.ply" "$modeldir/point_cloud/iteration_$ITERATIONS"
fi
