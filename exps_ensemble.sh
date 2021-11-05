#!/bin/sh

sbatch ./infer.sh 224 30 0.25
sbatch ./infer.sh 224 30 0.12
sbatch ./infer.sh 224 30 0.5
sbatch ./infer.sh 224 30 1.0
sbatch ./infer.sh 224 128 0.25
sbatch ./infer.sh 224 128 0.12
sbatch ./infer.sh 224 128 0.5
sbatch ./infer.sh 224 128 1.0

