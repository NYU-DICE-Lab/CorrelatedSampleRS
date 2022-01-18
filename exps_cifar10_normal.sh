#!/bin/sh

#sbatch ./infer_wo_max.sh 32 1 0.25 
#sbatch ./infer_wo_max.sh 32 1 0.12 
sbatch ./infer_wo_max.sh 32 1 0.50 
sbatch ./infer_wo_max.sh 32 1 1.00 
