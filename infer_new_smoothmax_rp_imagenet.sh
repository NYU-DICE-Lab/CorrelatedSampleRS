#!/bin/bash

#SBATCH --output=out_%A_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=47:59:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=certify_infer_minmax

PATCH_SIZE=$1
PATCH_STRIDE=$2
SIGMA=$3
RMODE=$4
MAXPATCHES=$5
START_IDX=$6

module purge

singularity exec --nv \
	--overlay /scratch/aaj458/singularity_containers/my_pytorch.ext3:ro \
	/scratch/aaj458/singularity_containers/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
	/bin/bash -c "source /ext3/env.sh; python infer_certify_pretrained_salman_uncorrelated.py imagenet -dpath /scratch/aaj458/data/ImageNet/val/ -mp salman_models/pretrained_models/imagenet/PGD_1step/imagenet/eps_512/resnet50/noise_$SIGMA/checkpoint.pth.tar -mt resnet50 -ni 500 -ps $PATCH_SIZE -pstr $PATCH_STRIDE -sigma $SIGMA --N0 100 --N 10000 --patch -o imagenet_SOTA/certify_results_salman_patchsmooth_smoothmean_randompatches_$MAXPATCHES/ --batch 128 -rm $RMODE -ns 256 -rp -np $MAXPATCHES -si $START_IDX"

#singularity exec --nv \
#	--overlay /scratch/aaj458/singularity_containers/my_pytorch.ext3:ro \
#	/scratch/aaj458/singularity_containers/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
#	/bin/bash -c "source /ext3/env.sh; python infer_certify_pretrained_salman.py imagenet -dpath /scratch/aaj458/data/ImageNet/val/ -mt resnet50 -mp salman_models/pretrained_models/imagenet/PGD_1step/imagenet/eps_512/resnet50/noise_$SIGMA/checkpoint.pth.tar -ni 500 -ps $PATCH_SIZE -pstr $PATCH_STRIDE -sigma $SIGMA --N0 100 --N 10000 -o imagenet_salman_SOTA/ --batch 128 -ns 224"
