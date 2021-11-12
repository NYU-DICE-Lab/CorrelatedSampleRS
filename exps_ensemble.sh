#!/bin/sh

# sbatch ./infer.sh 224 30 0.25
# sbatch ./infer.sh 224 30 0.12
# sbatch ./infer.sh 224 30 0.5
# sbatch ./infer.sh 224 30 1.0
# sbatch ./infer.sh 224 128 0.25
# sbatch ./infer.sh 224 128 0.12
# sbatch ./infer.sh 224 128 0.5
# sbatch ./infer.sh 224 128 1.0
sbatch ./infer.sh 224 60 0.25
sbatch ./infer.sh 224 60 0.12
sbatch ./infer.sh 224 60 0.5
sbatch ./infer.sh 224 60 1.0
sbatch ./infer.sh 224 15 0.25
sbatch ./infer.sh 224 15 0.12
sbatch ./infer.sh 224 15 0.5
sbatch ./infer.sh 224 15 1.0
sbatch ./infer.sh 224 90 0.25
sbatch ./infer.sh 224 90 0.12
sbatch ./infer.sh 224 90 0.5
sbatch ./infer.sh 224 90 1.0

