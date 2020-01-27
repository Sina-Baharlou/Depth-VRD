#!/usr/bin/env bash
# Train depth models using different configurations
N_SEEDS=8
N_EPOCHS=25
BATCH_SIZE=32
CLIP_GRAD=5
PREV_ITER=500
GPU_COUNT=1
FRCNN_CKPT="checkpoints/vg-faster-rcnn.tar"
SAVE_PATH="checkpoints"
DEPTH_MODEL="resnet18"

# -- *** TRAINING DEPTH MODEL (WITHOUT FUSION LAYER) *** --
if [[ $1 == "1.1" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING DEPTH MODEL (WITHOUT FUSION LAYER) | ADAM LR-4 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_depth \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-4 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/depth_${DEPTH_MODEL}_lr4_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "1.2" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING DEPTH MODEL (WITHOUT FUSION LAYER) | ADAM LR-5 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_depth \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-5 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/depth_${DEPTH_MODEL}_lr5_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "1.3" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING DEPTH MODEL (WITHOUT FUSION LAYER) | ADAM LR-6 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_depth \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-6 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/depth_${DEPTH_MODEL}_lr6_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -load_depth -depth_model ${DEPTH_MODEL}
    done


# -- *** TRAINING DEPTH MODEL UoBB (WITHOUT FUSION LAYER) *** --
elif [[ $1 == "2.1" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING DEPTH MODEL UoBB (WITHOUT FUSION LAYER) | ADAM LR-4 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_depth_union \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-4 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/depth_union_${DEPTH_MODEL}_lr4_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "2.2" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING DEPTH MODEL UoBB (WITHOUT FUSION LAYER) | ADAM LR-5 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_depth_union \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-5 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/depth_union_${DEPTH_MODEL}_lr5_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "2.3" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING DEPTH MODEL UoBB (WITHOUT FUSION LAYER) | ADAM LR-6 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_depth_union \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-6 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/depth_union_${DEPTH_MODEL}_lr6_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -load_depth -depth_model ${DEPTH_MODEL}
    done
fi