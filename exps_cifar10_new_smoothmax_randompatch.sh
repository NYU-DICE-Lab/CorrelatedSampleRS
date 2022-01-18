#!/bin/sh

sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 max 25
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 max 25
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.5 max 25
sbatch ./infer_new_smoothmax_rp.sh 32 1 1.0 max 25
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 max 50
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 max 50
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.5 max 50
sbatch ./infer_new_smoothmax_rp.sh 32 1 1.0 max 50
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 max 100
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 max 100
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.5 max 100
sbatch ./infer_new_smoothmax_rp.sh 32 1 1.0 max 100

