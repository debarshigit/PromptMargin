#!/bin/bash

DATASET=$1

if [ -z "$DATASET" ]; then
    echo "Please provide a dataset name."
    exit 1
fi

python train_test_files/train_test_cdfsl_${DATASET}.py