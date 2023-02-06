#!/bin/bash

#SBATCH --output=logs/out_%A_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=47:59:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=certify_infer_max_mean

PATCH_SIZE=$1
PATCH_STRIDE=$2
SIGMA=$3
RMODE=$4
MAXPATCHES=$5
START_IDX=$6
N=100000

module purge

singularity exec --nv \
	--overlay /scratch/aaj458/singularity_containers/my_pytorch.ext3:ro \
	/scratch/aaj458/singularity_containers/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
	/bin/bash -c "source /ext3/env.sh; python infer_certify_pretrained_salman_uncorrelated.py cifar10 -dpath /scratch/aaj458/data/Cifar10/ --alpha 0.001 -mp /scratch/aaj458/Projects/CorrelatedSampleRS/salman_models/pretrained_models/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_$SIGMA/checkpoint.pth.tar -mt resnet110 -ni 5000 -ps $PATCH_SIZE -pstr $PATCH_STRIDE -sigma $SIGMA --N0 100 --N $N --patch -o cifar10_ICLR_SOTA_upsample/certify_salman_smooth_randompatches_$MAXPATCHES_$RM/ --batch 400 -rm $RMODE -ns 32 -np $MAXPATCHES --normalize -si $START_IDX"

## Modified alpha acc. to Horwath
# singularity exec --nv \
# 	--overlay /scratch/aaj458/singularity_containers/my_pytorch.ext3:ro \
# 	/scratch/aaj458/singularity_containers/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
# 	/bin/bash -c "source /ext3/env.sh; python infer_certify_pretrained_salman_uncorrelated.py cifar10 -dpath /scratch/aaj458/data/Cifar10/ --alpha 0.01 -mp salman_models/pretrained_models/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_$SIGMA/checkpoint.pth.tar -mt resnet110 -ni 5000 -ps $PATCH_SIZE -pstr $PATCH_STRIDE -sigma $SIGMA --N0 100 --N $N --patch -o cifar10_ICLR_SOTA_4x100000_adjusted_001/certify_salman_smooth_randompatches_$MAXPATCHES_$RM/ --batch 400 -rm $RMODE -ns 36 -rp -np $MAXPATCHES --normalize -si $START_IDX"

# singularity exec --nv \
# 	--overlay /scratch/aaj458/singularity_containers/my_pytorch.ext3:ro \
# 	/scratch/aaj458/singularity_containers/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
# 	/bin/bash -c "source /ext3/env.sh; python infer_certify_pretrained_salman.py cifar10 -dpath /scratch/aaj458/data/Cifar10/ -mt resnet110 -mp salman_models/pretrained_models/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_$SIGMA/checkpoint.pth.tar -ni 10000 -ps $PATCH_SIZE -pstr $PATCH_STRIDE -sigma $SIGMA --N0 100 --N 400000 -o cifar10_SOTA_400k/ --batch 400 --normalize"
