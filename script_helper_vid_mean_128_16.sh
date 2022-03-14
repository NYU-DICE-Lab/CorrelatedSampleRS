#!/bin/bash

#SBATCH --output=out_%A_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=47:59:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=certify_ucf101_max

SIGMA=$1
MODEL_PATH=$2
RM=$3

SAMPLE_DURATION=16
CS=16
CSTR=16
SUBVIDEO_SIZE=128

module purge

singularity exec --nv \
	--overlay /scratch/mp5847/singularity_containers/overlay-50G-10M.ext3:ro \
	/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	/bin/bash -c "source /ext3/env.sh; python3 infer_certify_video.py UCF101 -o ucf101_$RM-subsize-video=$SUBVIDEO_SIZE-chunk-size=$CS/ -dpath /scratch/mp5847/UCF-101/ -sigma $SIGMA -mp $MODEL_PATH --frame_dir /scratch/mp5847/dataset/UCF-101_extracted_frames/ --annotation_path /home/mp5847/src/MARS/annotation/ucfTrainTestlist --sample_duration $SAMPLE_DURATION -cs $CS -cstr 16 -rs -ns 6 -rm $RM --N 100 --N0 1000 --batch 10 --only_RGB --subvideo_size $SUBVIDEO_SIZE"


