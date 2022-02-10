#!/bin/sh

# sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 max 25 9521 
# sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 max 25 9554 
# sbatch ./infer_new_smoothmax_rp.sh 32 1 0.50 max 25 9658
# sbatch ./infer_new_smoothmax_rp.sh 32 1 1.00 max 25 9630
# sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 max 50 4836
# sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 max 50 4853
# sbatch ./infer_new_smoothmax_rp.sh 32 1 0.50 max 50 4841
# sbatch ./infer_new_smoothmax_rp.sh 32 1 1.00 max 50 4839
# sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 max 100 2395
# sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 max 100 2396
# sbatch ./infer_new_smoothmax_rp.sh 32 1 0.50 max 100 2397
# sbatch ./infer_new_smoothmax_rp.sh 32 1 1.00 max 100 2412

sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 mean 25 0 
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 mean 25 0 
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.50 mean 25 0
sbatch ./infer_new_smoothmax_rp.sh 32 1 1.00 mean 25 0
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 mean 50 0
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 mean 50 0
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.50 mean 50 0
sbatch ./infer_new_smoothmax_rp.sh 32 1 1.00 mean 50 0
# sbatch ./infer_new_smoothmean_rp.sh 32 1 0.25 mean 100 0
# sbatch ./infer_new_smoothmean_rp.sh 32 1 0.12 mean 100 0
# sbatch ./infer_new_smoothmean_rp.sh 32 1 0.50 mean 100 0
# sbatch ./infer_new_smoothmean_rp.sh 32 1 1.00 mean 100 0