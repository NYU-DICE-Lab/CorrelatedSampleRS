#!/bin/sh

sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 max 25 3209  
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 max 25 3176 
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.50 max 25 3227
sbatch ./infer_new_smoothmax_rp.sh 32 1 1.00 max 25 3197
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 max 50 1604
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 max 50 1612
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.50 max 50 1611
sbatch ./infer_new_smoothmax_rp.sh 32 1 1.00 max 50 1613
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.25 max 100 798
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.12 max 100 805
sbatch ./infer_new_smoothmax_rp.sh 32 1 0.50 max 100 796
sbatch ./infer_new_smoothmax_rp.sh 32 1 1.00 max 100 810

