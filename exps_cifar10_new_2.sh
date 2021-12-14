#!/bin/sh

sbatch ./infer_new_2.sh 32 4 0.25 max
sbatch ./infer_new_2.sh 32 4 0.12 max
sbatch ./infer_new_2.sh 32 4 0.5 max
sbatch ./infer_new_2.sh 32 4 1.0 max
sbatch ./infer_new_2.sh 32 2 0.25 max
sbatch ./infer_new_2.sh 32 2 0.12 max
sbatch ./infer_new_2.sh 32 2 0.5 max
sbatch ./infer_new_2.sh 32 2 1.0 max
sbatch ./infer_new_2.sh 32 4 0.25 min
sbatch ./infer_new_2.sh 32 4 0.12 min
sbatch ./infer_new_2.sh 32 4 0.5 min
sbatch ./infer_new_2.sh 32 4 1.0 min
sbatch ./infer_new_2.sh 32 2 0.25 min
sbatch ./infer_new_2.sh 32 2 0.12 min
sbatch ./infer_new_2.sh 32 2 0.5 min
sbatch ./infer_new_2.sh 32 2 1.0 min
sbatch ./infer_new_2.sh 32 4 0.25 mean
sbatch ./infer_new_2.sh 32 4 0.12 mean
sbatch ./infer_new_2.sh 32 4 0.5 mean
sbatch ./infer_new_2.sh 32 4 1.0 mean
sbatch ./infer_new_2.sh 32 2 0.25 mean
sbatch ./infer_new_2.sh 32 2 0.12 mean
sbatch ./infer_new_2.sh 32 2 0.5 mean
sbatch ./infer_new_2.sh 32 2 1.0 mean

