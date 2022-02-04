#!/bin/sh

sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 0.25 max 10 0  
sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 0.12 max 10 0 
sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 0.50 max 10 0
sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 1.00 max 10 0
sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 0.25 max 25 0
sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 0.12 max 25 0
sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 0.50 max 25 0
sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 1.00 max 25 0
sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 0.25 max 50 0
sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 0.12 max 50 0
sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 0.50 max 50 0
sbatch ./infer_new_smoothmax_rp_imagenet.sh 224 1 1.00 max 50 0

