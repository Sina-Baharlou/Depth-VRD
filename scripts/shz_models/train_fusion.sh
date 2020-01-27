#!/usr/bin/env bash
# Train fusion models using different configurations
N_SEEDS=8
N_EPOCHS=25
BATCH_SIZE=16
CLIP_GRAD=5
PREV_ITER=500
GPU_COUNT=1
FRCNN_CKPT="checkpoints/vg-faster-rcnn.tar"
SAVE_PATH="checkpoints"
DEPTH_MODEL="resnet18"

# -- *** TRAINING FUSION MODEL (V,D) FROM SCRATCH *** --
if [[ $1 == "1.1" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (V,D) FROM SCRATCH | ADAM LR-4 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-4 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/fusion_vd_s_lr4_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features vd -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "1.2" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (V,D) FROM SCRATCH | ADAM LR-5 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-5 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/fusion_vd_s_lr5_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features vd -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "1.3" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (V,D) FROM SCRATCH | ADAM LR-6 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-6 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/fusion_vd_s_lr6_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features vd -load_depth -depth_model ${DEPTH_MODEL}
    done


# -- *** TRAINING FUSION MODEL (V,D) FROZEN*** --
elif [[ $1 == "2.1" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (V,D) FROZEN | ADAM LR-4 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-4 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/fusion_vd_lr4_seed_${i} \
        -ckpt [RGB_CHECKPOINT] -extra_ckpt [DEPTH_CHECKPOINT] \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features vd -frozen_features vd -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "2.2" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (V,D) FROZEN | ADAM LR-5 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-5 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/fusion_vd_lr5_seed_${i} \
        -ckpt [RGB_CHECKPOINT] -extra_ckpt [DEPTH_CHECKPOINT] \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features vd -frozen_features vd -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "2.3" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (V,D) FROZEN | ADAM LR-6 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-6 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/fusion_vd_lr6_seed_${i} \
        -ckpt [RGB_CHECKPOINT] -extra_ckpt [DEPTH_CHECKPOINT] \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features vd -frozen_features vd -load_depth -depth_model ${DEPTH_MODEL}
    done


# -- *** TRAINING FUSION MODEL (L,V,D) FROM SCRATCH *** --
elif [[ $1 == "3.1" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,V,D) FROM SCRATCH | ADAM LR-4 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-4 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/fusion_lvd_s_lr4_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lvd -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "3.2" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,V,D) FROM SCRATCH | ADAM LR-5 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-5 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/fusion_lvd_s_lr5_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lvd -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "3.3" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,V,D) FROM SCRATCH | ADAM LR-6 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-6 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/fusion_lvd_s_lr6_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lvd -load_depth -depth_model ${DEPTH_MODEL}
    done


# -- *** TRAINING FUSION MODEL (L,V,D) FROZEN*** --
elif [[ $1 == "4.1" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,V,D) FROZEN | ADAM LR-4 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-4 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/fusion_lvd_lr4_seed_${i} \
        -ckpt [RGB_CHECKPOINT] -extra_ckpt [DEPTH_CHECKPOINT] \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lvd -frozen_features lvd -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "4.2" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,V,D) FROZEN | ADAM LR-5 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-5 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/fusion_lvd_lr5_seed_${i} \
        -ckpt [RGB_CHECKPOINT] -extra_ckpt [DEPTH_CHECKPOINT] \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lvd -frozen_features lvd -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "4.3" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,V,D) FROZEN | ADAM LR-6 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-6 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/fusion_lvd_lr6_seed_${i} \
        -ckpt [RGB_CHECKPOINT] -extra_ckpt [DEPTH_CHECKPOINT] \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lvd -frozen_features lvd -load_depth -depth_model ${DEPTH_MODEL}
    done


# -- *** TRAINING FUSION MODEL (L,C,D) FROM SCRATCH *** --
elif [[ $1 == "5.1" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,D) FROM SCRATCH | ADAM LR-4 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-4 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/fusion_lcd_s_lr4_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcd -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "5.2" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,D) FROM SCRATCH | ADAM LR-5 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-5 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/fusion_lcd_s_lr5_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcd -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "5.3" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,D) FROM SCRATCH | ADAM LR-6 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-6 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/fusion_lcd_s_lr6_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcd -load_depth -depth_model ${DEPTH_MODEL}
    done


# -- *** TRAINING FUSION MODEL (L,C,D) FROZEN*** --
elif [[ $1 == "6.1" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,D) FROZEN | ADAM LR-4 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-4 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/fusion_lcd_lr4_seed_${i} \
        -ckpt [LC_CHECKPOINT] -extra_ckpt [DEPTH_CHECKPOINT] \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcd -frozen_features lcd -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "6.2" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,D) FROZEN | ADAM LR-5 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-5 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/fusion_lcd_lr5_seed_${i} \
        -ckpt [LC_CHECKPOINT] -extra_ckpt [DEPTH_CHECKPOINT] \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcd -frozen_features lcd -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "6.3" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,D) FROZEN | ADAM LR-6 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-6 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/fusion_lcd_lr6_seed_${i} \
        -ckpt [LC_CHECKPOINT] -extra_ckpt [DEPTH_CHECKPOINT] \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcd -frozen_features lcd -load_depth -depth_model ${DEPTH_MODEL}
    done


# -- *** TRAINING FUSION MODEL (L,C,V) FROM SCRATCH *** --
elif [[ $1 == "7.1" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,V) FROM SCRATCH | ADAM LR-4 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-4 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/fusion_lcv_s_lr4_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcv
    done
elif [[ $1 == "7.2" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,V) FROM SCRATCH | ADAM LR-5 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-5 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/fusion_lcv_s_lr5_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcv
    done
elif [[ $1 == "7.3" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,V) FROM SCRATCH | ADAM LR-6 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-6 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/fusion_lcv_s_lr6_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcd
    done


# -- *** TRAINING FUSION MODEL (L,C,V,D) FROM SCRATCH *** --
elif [[ $1 == "8.1" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,V,D) FROM SCRATCH | ADAM LR-4 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-4 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/fusion_lcvd_s_lr4_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcvd -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "8.2" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,V,D) FROM SCRATCH | ADAM LR-5 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-5 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/fusion_lcvd_s_lr5_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcvd -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "8.3" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,V,D) FROM SCRATCH | ADAM LR-6 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-6 -clip ${CLIP_GRAD} \
        -ckpt ${FRCNN_CKPT} -save_dir ${SAVE_PATH}/fusion_lcvd_s_lr6_seed_${i} \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcvd -load_depth -depth_model ${DEPTH_MODEL}
    done


# -- *** TRAINING FUSION MODEL (L,C,V,D) FROZEN*** --
elif [[ $1 == "9.1" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,V,D) FROZEN | ADAM LR-4 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-4 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/fusion_lcvd_lr4_seed_${i} \
        -ckpt [RGB_CHECKPOINT] -extra_ckpt [DEPTH_CHECKPOINT] \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcvd -frozen_features lcvd -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "9.2" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,V,D) FROZEN | ADAM LR-5 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-5 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/fusion_lcvd_lr5_seed_${i} \
        -ckpt [RGB_CHECKPOINT] -extra_ckpt [DEPTH_CHECKPOINT] \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcvd -frozen_features lcvd -load_depth -depth_model ${DEPTH_MODEL}
    done
elif [[ $1 == "9.3" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,V,D) FROZEN | ADAM LR-6 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-6 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/fusion_lcvd_lr6_seed_${i} \
        -ckpt [RGB_CHECKPOINT] -extra_ckpt [DEPTH_CHECKPOINT] \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcvd -frozen_features lcvd -load_depth -depth_model ${DEPTH_MODEL}
    done


# -- *** TRAINING FUSION MODEL (L,C,V,D) FINETUNE*** --
elif [[ $1 == "10.1" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,V,D) FINETUNE | ADAM LR-4 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-4 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/fusion_lcvd_ft_lr4_seed_${i} \
        -ckpt [YOUR_CHECKPOINT] \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcvd -load_depth -depth_model ${DEPTH_MODEL} -keep_weights
    done
elif [[ $1 == "10.2" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,V,D) FINETUNE | ADAM LR-5 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-5 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/fusion_lcvd_ft_lr5_seed_${i} \
        -ckpt [YOUR_CHECKPOINT] \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcvd -load_depth -depth_model ${DEPTH_MODEL} -keep_weights
    done
elif [[ $1 == "10.3" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,V,D) FINETUNE | ADAM LR-6 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-6 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/fusion_lcvd_ft_lr6_seed_${i} \
        -ckpt [YOUR_CHECKPOINT] \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcvd -load_depth -depth_model ${DEPTH_MODEL} -keep_weights
    done
elif [[ $1 == "10.4" ]]; then
    for i in $(seq 0 $((N_SEEDS - 1)))
    do
    echo "TRAINING FUSION MODEL (L,C,V,D) FINETUNE | ADAM LR-7 | SEED: $i"
    python models/train_rels.py -m predcls -model shz_fusion -hidden_dim 4096 \
        -b ${BATCH_SIZE} -nepoch ${N_EPOCHS} -adam -lr 1e-7 -clip ${CLIP_GRAD} \
        -save_dir ${SAVE_PATH}/fusion_lcvd_ft_lr7_seed_${i} \
        -ckpt [YOUR_CHECKPOINT] \
        -p ${PREV_ITER} -tensorboard_ex  -ngpu ${GPU_COUNT} -rnd_seed ${i} \
        -active_features lcvd -load_depth -depth_model ${DEPTH_MODEL} -keep_weights
    done
fi