#!/usr/bin/env bash
# Train Autoencoder using different configurations (DEPRECATED)
N_EPOCHS=25
BATCH_SIZE=16
CLIP_GRAD=5
PREV_ITER=500
GPU_COUNT=1
SAVE_PATH="checkpoints"
DEPTH_MODEL="alexnet"

# -- *** TRAIN DEPTH AUTOENCODER WITH ADAM *** --
if [[ $1 == "1" ]]; then
    echo "TRAINING ALEXNET DEPTH AUTOENCODER ADAM-LR3"
    python models/train_ae.py  -b ${BATCH_SIZE} -clip ${CLIP_GRAD} -p ${PREV_ITER} -lr 1e-3 \
        -ngpu ${GPU_COUNT}  -save_dir checkpoints/ae-alexnet-adam-lr3 -nepoch ${N_EPOCHS} \
        -depth_model ${DEPTH_MODEL} -adam -tensorboard_ex
elif [[ $1 == "2" ]]; then
    echo "TRAINING ALEXNET DEPTH AUTOENCODER ADAM-LR4"
    python models/train_ae.py  -b ${BATCH_SIZE} -clip ${CLIP_GRAD} -p ${PREV_ITER} -lr 1e-4 \
        -ngpu ${GPU_COUNT}  -save_dir checkpoints/ae-alexnet-adam-lr4 -nepoch ${N_EPOCHS} \
        -depth_model ${DEPTH_MODEL} -adam -tensorboard_ex
elif [[ $1 == "3" ]]; then
    echo "TRAINING ALEXNET DEPTH AUTOENCODER ADAM-LR5"
    python models/train_ae.py  -b ${BATCH_SIZE} -clip ${CLIP_GRAD} -p ${PREV_ITER} -lr 1e-5 \
        -ngpu ${GPU_COUNT}  -save_dir checkpoints/ae-alexnet-adam-lr5 -nepoch ${N_EPOCHS} \
        -depth_model ${DEPTH_MODEL} -adam -tensorboard_ex
elif [[ $1 == "4" ]]; then
    echo "TRAINING ALEXNET DEPTH AUTOENCODER ADAM-LR6"
    python models/train_ae.py  -b ${BATCH_SIZE} -clip ${CLIP_GRAD} -p ${PREV_ITER} -lr 1e-6 \
        -ngpu ${GPU_COUNT}  -save_dir checkpoints/ae-alexnet-adam-lr6 -nepoch ${N_EPOCHS} \
        -depth_model ${DEPTH_MODEL} -adam -tensorboard_ex

# -- *** TRAIN DEPTH AUTOENCODER WITH SGD *** --
elif [[ $1 == "5" ]]; then
    echo "TRAINING ALEXNET DEPTH AUTOENCODER SGD-LR3"
    python models/train_ae.py  -b ${BATCH_SIZE} -clip ${CLIP_GRAD} -p ${PREV_ITER} -lr 1e-3 \
        -ngpu ${GPU_COUNT}  -save_dir checkpoints/ae-alexnet-sgd-lr3 -nepoch ${N_EPOCHS} \
        -depth_model ${DEPTH_MODEL} -tensorboard_ex
elif [[ $1 == "6" ]]; then
    echo "TRAINING ALEXNET DEPTH AUTOENCODER SGD-LR4"
    python models/train_ae.py  -b ${BATCH_SIZE} -clip ${CLIP_GRAD} -p ${PREV_ITER} -lr 1e-4 \
        -ngpu ${GPU_COUNT}  -save_dir checkpoints/ae-alexnet-sgd-lr4 -nepoch ${N_EPOCHS} \
        -depth_model ${DEPTH_MODEL} -tensorboard_ex
elif [[ $1 == "7" ]]; then
    echo "TRAINING ALEXNET DEPTH AUTOENCODER SGD-LR5"
    python models/train_ae.py  -b ${BATCH_SIZE} -clip ${CLIP_GRAD} -p ${PREV_ITER} -lr 1e-5 \
        -ngpu ${GPU_COUNT}  -save_dir checkpoints/ae-alexnet-sgd-lr5 -nepoch ${N_EPOCHS} \
        -depth_model ${DEPTH_MODEL} -tensorboard_ex
elif [[ $1 == "8" ]]; then
    echo "TRAINING ALEXNET DEPTH AUTOENCODER SGD-LR6"
    python models/train_ae.py  -b ${BATCH_SIZE} -clip ${CLIP_GRAD} -p ${PREV_ITER} -lr 1e-6 \
        -ngpu ${GPU_COUNT}  -save_dir checkpoints/ae-alexnet-sgd-lr6 -nepoch 25 \
        -depth_model ${DEPTH_MODEL} -tensorboard_ex
fi