#!/bin/bash

#SBATCH -t 08:00:00
#SBATCH -p GPU-AI
#SBATCH --gres=gpu:volta16:1

source activate
conda activate keras-retinanet4
cd ~/scratch/stamps/keras-retinanet

time keras_retinanet/bin/train.py --batch-size=BATCH_SIZE --lr=LEARNING_RATE --epochs=EPOCHS --steps=STEPS \
--tensorboard-dir=/pylon5/hm5fp1p/results/campaign0/stamps_hyper_results/hyperHYPER_N/tensorboard/ \
--weights snapshots/resnet50_coco_best_v2.1.0.h5 \
--snapshot-path /pylon5/hm5fp1p/results/campaign0/stamps_hyper_results/hyperHYPER_N/snapshots \
coco ~/scratch/stamps/cross-validation/SPLIT/coco
