#!/bin/sh

sbatch ./salman_infer.sh 0 0 0.12
sbatch ./salman_infer.sh 0 0 0.25
sbatch ./salman_infer.sh 0 0 0.5
sbatch ./salman_infer.sh 0 0 1.0

