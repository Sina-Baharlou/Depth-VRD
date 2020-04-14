#!/usr/bin/env bash
# Evaluate trained models
BATCH_SIZE=16
GPU_COUNT=1
DEPTH_MODEL="resnet18"

# -- *** EVALUATE ISOLATED DEPTH MODEL (WITHOUT FUSION LAYER, Ours-d) *** --
if [[ $1 == "1" ]]; then
    echo "EVALUATING ISOLATED DEPTH MODEL (WITHOUT FUSION LAYER, Ours-d)"
    python models/eval_rels.py -m predcls -model shz_depth \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -load_depth -depth_model ${DEPTH_MODEL} -test

# -- *** EVALUATE ISOLATED DEPTH MODEL UoBB (WITHOUT FUSION LAYER) *** --
elif [[ $1 == "2" ]]; then
    echo "EVALUATING ISOLATED DEPTH MODEL UoBB (WITHOUT FUSION LAYER)"
    python models/eval_rels.py -m predcls -model shz_depth_union \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -load_depth -depth_model ${DEPTH_MODEL} -test

# -- *** EVALUATE DEPTH MODEL (Ours-d) *** --
elif [[ $1 == "3" ]]; then
    echo "EVALUATING DEPTH MODEL (WITH FUSION LAYER, Ours-d)"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features d -load_depth -depth_model ${DEPTH_MODEL} -test

# -- *** EVALUATE CLASS MODEL (Ours-c) *** --
elif [[ $1 == "4" ]]; then
    echo "EVALUATING CLASS MODEL (Ours-c)"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 128 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features c -test

# -- *** EVALUATE VISUAL MODEL (Ours-v) *** --
elif [[ $1 == "5" ]]; then
    echo "EVALUATING VISUAL MODEL (Ours-v)"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features v -test

# -- *** EVALUATE LOCATION MODEL (Ours-l) *** --
elif [[ $1 == "6" ]]; then
    echo "EVALUATING LOCATION MODEL (Ours-l)"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 128 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features l -test

# -- *** EVALUATE FUSION MODEL (Ours-v,d) *** --
elif [[ $1 == "7" ]]; then
    echo "EVALUATING FUSION MODEL (Ours-v,d) "
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features vd -load_depth -depth_model ${DEPTH_MODEL} -test

# -- *** EVALUATE FUSION MODEL (Ours-l,v,d) *** --
elif [[ $1 == "8" ]]; then
    echo "EVALUATING FUSION MODEL (Ours-l,v,d)"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features lvd -load_depth -depth_model ${DEPTH_MODEL} -test

# -- *** EVALUATE FUSION MODEL (Ours-l,c,d) *** --
elif [[ $1 == "9" ]]; then
    echo "EVALUATING FUSION MODEL (Ours-l,c,d)"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features lcd -load_depth -depth_model ${DEPTH_MODEL} -test

# -- *** EVALUATE FUSION MODEL (Ours-l,c,v) *** --
elif [[ $1 == "10" ]]; then
    echo "EVALUATING FUSION MODEL (Ours-l,c,v)"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features lcv -test

# -- *** EVALUATE FUSION MODEL (Ours-l,c,v,d) *** --
elif [[ $1 == "11" ]]; then
    echo "EVALUATING FUSION MODEL (Ours-l,c,v,d)"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt [YOUR_CHECKPOINT] \
        -active_features lcvd -load_depth -depth_model ${DEPTH_MODEL} -test

# -- *** EVALUATE FUSION MODEL (Ours-l,c,v,d) WITH OUR CHECKPOINT *** --
elif [[ $1 == "12" ]]; then
    echo "EVALUATING FUSION MODEL (Ours-l,c,v,d) WITH OUR CHECKPOINT"
    python models/eval_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -ngpu ${GPU_COUNT} \
        -ckpt "checkpoints/vgrel-lcvd.tar" \
        -active_features lcvd -load_depth -depth_model resnet18 -test
fi