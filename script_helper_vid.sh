SIGMA=0.5
SAMPLE_DURATION=8
CS=8
CSTR=8
SUBVIDEO_SIZE=64
MODEL_PATH=/home/mp5847/src/MARS/results/PreKin_UCF101_1_RGB_train_batch64_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx0_varLR114_sigma=0.5.pth

python3 infer_certify_video.py UCF101 -o vidtest/ -dpath /scratch/mp5847/UCF-101/ -sigma $SIGMA -mp $MODEL_PATH --frame_dir /scratch/mp5847/dataset/UCF-101_extracted_frames/ --annotation_path /home/mp5847/src/MARS/annotation/ucfTrainTestlist --sample_duration $SAMPLE_DURATION -cs $CS -cstr $CSTR -rs -svs $SUBVIDEO_SIZE -ns 6 --N 100 --N0 1000 --batch 10 --only_RGB --basers
