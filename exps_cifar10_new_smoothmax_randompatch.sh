#!/bin/sh

sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 max 25 9521 
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 max 25 9554 
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.50 max 25 9658
sbatch ./infer_new_smoothmax_rp.sh 32 1 1.00 max 25 9630
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 max 50 4836
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 max 50 4853
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.50 max 50 4841
sbatch ./infer_new_smoothmax_rp.sh 32 1 1.00 max 50 4839
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 max 100 2395
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 max 100 2396
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.50 max 100 2397
sbatch ./infer_new_smoothmax_rp.sh 32 1 1.00 max 100 2412

