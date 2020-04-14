#!/usr/bin/env bash
# Train individual models using different configurations
N_SEEDS=8
N_EPOCHS=30
BATCH_SIZE=32
CLIP_GRAD=5
PREV_ITER=500
GPU_COUNT=1
FRCNN_CKPT="checkpoints/vg-faster-rcnn.tar"
SAVE_PATH="checkpoints"
DEPTH_MODEL="resnet18"

# -- *** TRAIN INDIVIDUAL MODEL (Ours-c) *** --
if [[ $1 == "1.1" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING INDIVIDUAL MODEL (Ours-c) | ADAM LR-4 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 128 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-4 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/indv_c_lr4_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features c
    done
elif [[ $1 == "1.2" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING INDIVIDUAL MODEL (Ours-c) | ADAM LR-5 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 128 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-5 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/indv_c_lr5_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features c
    done
elif [[ $1 == "1.3" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING INDIVIDUAL MODEL (Ours-c) | ADAM LR-6 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 128 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-6 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/indv_c_lr6_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features c
    done


# -- *** TRAIN INDIVIDUAL MODEL (Ours-v) *** --
elif [[ $1 == "2.1" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING INDIVIDUAL MODEL (Ours-v) | ADAM LR-4 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-4 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/indv_v_lr4_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features v
    done
elif [[ $1 == "2.2" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING INDIVIDUAL MODEL (Ours-v) | ADAM LR-5 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-5 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/indv_v_lr5_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features v
    done
elif [[ $1 == "2.3" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING INDIVIDUAL MODEL (Ours-v) | ADAM LR-6 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-6 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/indv_v_lr6s_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features v
    done


# -- *** TRAIN INDIVIDUAL MODEL (Ours-l) *** --
elif [[ $1 == "3.1" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING INDIVIDUAL MODEL (Ours-l) | ADAM LR-4 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 128 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-4 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/indv_l_lr4_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features l
    done
elif [[ $1 == "3.2" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING INDIVIDUAL MODEL (Ours-l) | ADAM LR-5 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 128 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-5 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/indv_l_lr5_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features l
    done
elif [[ $1 == "3.3" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING INDIVIDUAL MODEL (Ours-l) | ADAM LR-6 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 128 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-6 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/indv_l_lr6_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features l
    done


# -- *** TRAIN INDIVIDUAL MODEL (DEPTH) *** --
elif [[ $1 == "4.1" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING INDIVIDUAL MODEL (Ours-d) | ADAM LR-4 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-4 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/indv_d_${DEPTH_MODEL}_lr4_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features d -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "4.2" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING INDIVIDUAL MODEL (Ours-d) | ADAM LR-5 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-5 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/indv_d_${DEPTH_MODEL}_lr5_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features d -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "4.3" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING INDIVIDUAL MODEL (Ours-d) | ADAM LR-6 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-6 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/indv_d_${DEPTH_MODEL}_lr6_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features d -load_depth -depth_model ${DEPTH_MODEL}
    done
fi