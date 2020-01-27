#!/usr/bin/env bash

# Train Autoencoder using different configurations

# export CUDA_VISIBLE_DEVICES=$1
echo $1

### --- TRAIN WITH ADAM -- ###
if [[ $1 == "0" ]]; then
    echo "TRAINING ALEXNET DEPTH AUTOENCODER ADAM-LR3"
    python models/train_ae.py  -b 16 -clip 5 -p 500 -lr 1e-3 \
        -ngpu 1  -save_dir checkpoints/ae-alexnet-adam-lr3 -nepoch 25 \
        -fusion_mode fusion -depth_model alexnet -adam -tensorboard_ex

elif [[ $1 == "1" ]]; then
    echo "TRAINING ALEXNET DEPTH AUTOENCODER ADAM-LR4"
    python models/train_ae.py  -b 16 -clip 5 -p 500 -lr 1e-4 \
        -ngpu 1  -save_dir checkpoints/ae-alexnet-adam-lr4 -nepoch 25 \
        -fusion_mode fusion -depth_model alexnet -adam -tensorboard_ex

elif [[ $1 == "2" ]]; then
    echo "TRAINING ALEXNET DEPTH AUTOENCODER ADAM-LR5"
    python models/train_ae.py  -b 16 -clip 5 -p 500 -lr 1e-5 \
        -ngpu 1  -save_dir checkpoints/ae-alexnet-adam-lr5 -nepoch 25 \
        -fusion_mode fusion -depth_model alexnet -adam -tensorboard_ex

elif [[ $1 == "3" ]]; then
    echo "TRAINING ALEXNET DEPTH AUTOENCODER ADAM-LR6"
    python models/train_ae.py  -b 16 -clip 5 -p 500 -lr 1e-6 \
        -ngpu 1  -save_dir checkpoints/ae-alexnet-adam-lr6 -nepoch 25 \
        -fusion_mode fusion -depth_model alexnet -adam -tensorboard_ex

### --- TRAIN WITH SGD -- ###
elif [[ $1 == "4" ]]; then
    echo "TRAINING ALEXNET DEPTH AUTOENCODER SGD-LR3"
    python models/train_ae.py  -b 16 -clip 5 -p 500 -lr 1e-3 \
        -ngpu 1  -save_dir checkpoints/ae-alexnet-sgd-lr3 -nepoch 25 \
        -fusion_mode fusion -depth_model alexnet -tensorboard_ex

elif [[ $1 == "5" ]]; then
    echo "TRAINING ALEXNET DEPTH AUTOENCODER SGD-LR4"
    python models/train_ae.py  -b 16 -clip 5 -p 500 -lr 1e-4 \
        -ngpu 1  -save_dir checkpoints/ae-alexnet-sgd-lr4 -nepoch 25 \
        -fusion_mode fusion -depth_model alexnet -tensorboard_ex

elif [[ $1 == "6" ]]; then
    echo "TRAINING ALEXNET DEPTH AUTOENCODER SGD-LR5"
    python models/train_ae.py  -b 16 -clip 5 -p 500 -lr 1e-5 \
        -ngpu 1  -save_dir checkpoints/ae-alexnet-sgd-lr5 -nepoch 25 \
        -fusion_mode fusion -depth_model alexnet -tensorboard_ex

elif [[ $1 == "7" ]]; then
    echo "TRAINING ALEXNET DEPTH AUTOENCODER SGD-LR6"
    python models/train_ae.py  -b 16 -clip 5 -p 500 -lr 1e-6 \
        -ngpu 1  -save_dir checkpoints/ae-alexnet-sgd-lr6 -nepoch 25 \
        -fusion_mode fusion -depth_model alexnet -tensorboard_ex
fi