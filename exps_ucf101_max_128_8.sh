#!/bin/sh

sbatch ./script_helper_vid_max_128_8.sh 0.25 /home/mp5847/src/MARS/results/PreKin_UCF101_1_RGB_train_batch64_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx0_varLR114_sigma=0.25.pth max
sbatch ./script_helper_vid_max_128_8.sh 0.5 /home/mp5847/src/MARS/results/PreKin_UCF101_1_RGB_train_batch64_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx0_varLR114_sigma=0.5.pth max
sbatch ./script_helper_vid_max_128_8.sh 1.0 /home/mp5847/src/MARS/results/PreKin_UCF101_1_RGB_train_batch64_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx0_varLR214_sigma=1.0.pth max