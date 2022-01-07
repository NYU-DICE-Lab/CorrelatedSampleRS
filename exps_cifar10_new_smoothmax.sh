#!/bin/sh

# sbatch ./infer_new_smoothmax.sh 32 4 0.25 max
# sbatch ./infer_new_smoothmax.sh 32 4 0.12 max
# sbatch ./infer_new_smoothmax.sh 32 4 0.5 max
# sbatch ./infer_new_smoothmax.sh 32 4 1.0 max
# sbatch ./infer_new_smoothmax.sh 32 2 0.25 max
# sbatch ./infer_new_smoothmax.sh 32 2 0.12 max
# sbatch ./infer_new_smoothmax.sh 32 2 0.5 max
# sbatch ./infer_new_smoothmax.sh 32 2 1.0 max
sbatch ./infer_new_smoothmax.sh 32 1 0.25 max
sbatch ./infer_new_smoothmax.sh 32 1 0.12 max
sbatch ./infer_new_smoothmax.sh 32 1 0.5 max
sbatch ./infer_new_smoothmax.sh 32 1 1.0 max
