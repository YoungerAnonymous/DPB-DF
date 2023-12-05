#!/bin/bash
datapath=data/$1
mode=${2:-train}
savepath=${3:-results}

if [ $mode == train ]
then
    # Train
    python run_dpbdf.py --gpu 0 --seed 0 --base_class 1 --incre_class 1 --save_model --save_results $savepath \
    dpb_df --k1 5 --k2 40 --dsr_n 0.1 --dsr_d 0.5 \
    dataset --resize 256 --imagesize 224 --batch_size 32 --train_val_split 1.0 $1 $datapath
elif [ $mode == test ]
then
    # Test
    python run_dpbdf.py --gpu 0 --seed 0 --base_class 1 --incre_class 1 --save_model --save_results $savepath \
    dpb_df --load_path $savepath/$1/checkpoints \
    dataset --resize 256 --imagesize 224 --batch_size 32 --train_val_split 1.0 $1 $datapath
else
    echo "Error: 'mode' argument should be in 'train' or 'test'."
fi
