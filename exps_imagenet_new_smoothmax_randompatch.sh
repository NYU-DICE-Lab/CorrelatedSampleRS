#!/bin/sh

# sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 0.25 max 4 0  
# sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 0.50 max 4 0
# sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 1.00 max 4 0
# sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 0.25 max 8 0
# sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 0.50 max 8 0
# sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 1.00 max 8 0
# sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 0.25 max 16 0
# sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 0.50 max 16 0
# sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 1.00 max 16 0
sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 0.25 mean 16 0
sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 0.50 mean 16 0
sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 1.00 mean 16 0
sbatch ./infer_new_smoothmax_rp_imagenet_ddn.sh 224 1 0.25 mean 16 0
sbatch ./infer_new_smoothmax_rp_imagenet_ddn.sh 224 1 0.50 mean 16 0
sbatch ./infer_new_smoothmax_rp_imagenet_ddn.sh 224 1 1.00 mean 16 0




