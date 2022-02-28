#!/bin/sh

# sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 max 25 0 
# sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 max 25 0 
# sbatch ./infer_new_smoothmax_rp.sh 32 1 0.50 max 25 0
# sbatch ./infer_new_smoothmax_rp.sh 32 1 1.00 max 25 0
# sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 mean 25 0 
# sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 mean 25 0 
# sbatch ./infer_new_smoothmax_rp.sh 32 1 0.50 mean 25 0
# sbatch ./infer_new_smoothmax_rp.sh 32 1 1.00 mean 25 0

sbatch ./infer_new_smoothmax_uniform.sh 32 1 0.25 max 25 0 
# sbatch ./infer_new_smoothmax_uniform.sh 32 1 0.12 max 25 0 
# sbatch ./infer_new_smoothmax_uniform.sh 32 1 0.50 max 25 0
# sbatch ./infer_new_smoothmax_uniform.sh 32 1 1.00 max 25 0
# sbatch ./infer_new_smoothmax_uniform.sh 32 1 0.25 mean 25 0 
# sbatch ./infer_new_smoothmax_uniform.sh 32 1 0.12 mean 25 0 
# sbatch ./infer_new_smoothmax_uniform.sh 32 1 0.50 mean 25 0
# sbatch ./infer_new_smoothmax_uniform.sh 32 1 1.00 mean 25 0
#sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 mean 25 0 
#sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 mean 25 0 
#sbatch ./infer_new_smoothmax_rp.sh 32 1 0.50 mean 25 0
#sbatch ./infer_new_smoothmax_rp.sh 32 1 1.00 mean 25 0
#sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 mean 25 0 
#sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 mean 25 0 
#sbatch ./infer_new_smoothmax_rp.sh 32 1 0.50 mean 25 0
#sbatch ./infer_new_smoothmax_rp.sh 32 1 1.00 mean 25 0
# sbatch ./infer_new_smoothmean_rp.sh 32 1 0.25 mean 100 0
# sbatch ./infer_new_smoothmean_rp.sh 32 1 0.12 mean 100 0
# sbatch ./infer_new_smoothmean_rp.sh 32 1 0.50 mean 100 0
# sbatch ./infer_new_smoothmean_rp.sh 32 1 1.00 mean 100 0