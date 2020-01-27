#!/usr/bin/env bash
# Evaluate fusion models using different configurations
BATCH_SIZE=16
GPU_COUNT=1
DEPTH_MODEL="resnet18"

# -- *** EVALUATING DEPTH MODEL (WITHOUT FUSION LAYER) *** --
if [[ $1 == "1" ]]; then
    echo "EVALUATING DEPTH MODEL (WITHOUT FUSION LAYER)"
    python models/eval_rels.py -m predcls -model shz_depth \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -load_depth -depth_model ${DEPTH_MODEL} -test

# -- *** EVALUATING DEPTH MODEL UoBB (WITHOUT FUSION LAYER) *** --
elif [[ $1 == "2" ]]; then
    echo "EVALUATING DEPTH MODEL UoBB (WITHOUT FUSION LAYER)"
    python models/eval_rels.py -m predcls -model shz_depth_union \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -load_depth -depth_model ${DEPTH_MODEL} -test

# -- *** EVALUATING INDIVIDUAL MODEL (CLASS) *** --
elif [[ $1 == "3" ]]; then
    echo "EVALUATING INDIVIDUAL MODEL (CLASS)"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 128 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features c -test

# -- *** EVALUATING INDIVIDUAL MODEL (VISUAL) *** --
elif [[ $1 == "4" ]]; then
    echo "EVALUATING INDIVIDUAL MODEL (VISUAL)"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features v -test

# -- *** EVALUATING INDIVIDUAL MODEL (LOCATION) *** --
elif [[ $1 == "5" ]]; then
    echo "EVALUATING INDIVIDUAL MODEL (LOCATION)"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 128 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features l -test

# -- *** EVALUATING INDIVIDUAL MODEL (DEPTH) *** --
elif [[ $1 == "6" ]]; then
    echo "EVALUATING INDIVIDUAL MODEL (DEPTH)"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features d -load_depth -depth_model ${DEPTH_MODEL} -test

# -- *** EVALUATING FUSION MODEL (V,D) FROM SCRATCH/FROZEN *** --
elif [[ $1 == "7" ]]; then
    echo "EVALUATING FUSION MODEL (V,D) FROM SCRATCH/FROZEN"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features vd -load_depth -depth_model ${DEPTH_MODEL} -test

# -- *** EVALUATING FUSION MODEL (L,V,D) FROM SCRATCH/FROZEN *** --
elif [[ $1 == "8" ]]; then
    echo "EVALUATING FUSION MODEL (L,V,D) FROM SCRATCH/FROZEN"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features lvd -load_depth -depth_model ${DEPTH_MODEL} -test

# -- *** EVALUATING FUSION MODEL (L,C,D) FROM SCRATCH/FROZEN *** --
elif [[ $1 == "9" ]]; then
    echo "EVALUATING FUSION MODEL (L,C,D) FROM SCRATCH/FROZEN"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features lcd -load_depth -depth_model ${DEPTH_MODEL} -test

# -- *** EVALUATING FUSION MODEL (L,C,V) FROM SCRATCH/FROZEN *** --
elif [[ $1 == "10" ]]; then
    echo "EVALUATING FUSION MODEL (L,C,V) FROM SCRATCH/FROZEN"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features lcv -test

# -- *** EVALUATING FUSION MODEL (L,C,V,D) FROM SCRATCH/FROZEN *** --
elif [[ $1 == "11" ]]; then
    echo "EVALUATING FUSION MODEL (L,C,V,D) FROM SCRATCH/FROZEN"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features lcvd -load_depth -depth_model ${DEPTH_MODEL} -test

# -- *** EVALUATING FUSION MODEL (L,C,V,D) WITH OUR CHECKPOINT *** --
elif [[ $1 == "12" ]]; then
    echo "EVALUATING FUSION MODEL (L,C,V,D) WITH OUR CHECKPOINT"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt "checkpoints/vgrel-lcvd.tar" \
        -active_features lcvd -load_depth -depth_model resnet18 -test
fi