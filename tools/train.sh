#!/usr/bin/env bash

FILE=$1
CONFIG=$2
GPUS=$3

python3 $FILE --config-file $CONFIG --num-gpus $GPUS ${@:4}