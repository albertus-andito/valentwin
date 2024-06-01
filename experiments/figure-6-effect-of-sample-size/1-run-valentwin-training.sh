#!/bin/bash

NUM_GPU=8
PORT_ID=$(expr $RANDOM + 1000)

declare -a training_sizes=("100" "200" "300" "400" "500")

declare -a alite_datasets=("25ksep11" "500spend" "1009ipopayments" "amo-ame" "chicago_parks" "cihr" "DCMS_NHM_NHM" "organogram-junior" "school_report" "stockport_contracts")
declare -a magellan_datasets=("academic_papers" "books" "cosmetics" "movies" "restaurants")

for size in "${training_sizes[@]}"; do
    for dataset in "${alite_datasets[@]}"; do
        torchrun --nproc-per-node=$NUM_GPU --rdzv-endpoint=localhost:$PORT_ID ../../scripts/train_valentwin.py \
          --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
          --train_file "../../data/alite/$dataset/contrastive-selective/$size/train.csv" \
          --validation_file "../../data/alite/$dataset/contrastive-selective/$size/val.csv" \
          --eval_file "../../data/alite/$dataset/contrastive-selective/$size/test.csv" \
          --output_dir "../result/valentwin-$dataset-n-$size-hn-10-selective-neg-lr-3e5-bs-512" \
          --num_train_epochs 3 \
          --per_device_train_batch_size 64 \
          --per_device_eval_batch_size 64 \
          --learning_rate 3e-5 \
          --max_seq_length 32 \
          --pooler_type cls \
          --overwrite_output_dir \
          --temp 0.05 \
          --do_train \
          --do_eval \
          --label_names [] \
          --logging_strategy epoch \
          --evaluation_strategy epoch \
          --save_strategy epoch \
          --metric_for_best_model accuracy \
          --load_best_model_at_end \
          --report_to wandb \
          --run_name "valentwin-$dataset-n-$size-hn-10-selective-neg-lr-3e5-bs-512" \
          --fp16 \
          --use_in_batch_instances_as_negatives \
          --restrictive_in_batch_negatives \
          "$@"
    done

    for dataset in "${magellan_datasets[@]}"; do
        torchrun --nproc-per-node=$NUM_GPU --rdzv-endpoint=localhost:$PORT_ID ../../scripts/train_valentwin.py \
          --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
          --train_file "../../data/magellan/$dataset/contrastive-selective/$size/train.csv" \
          --validation_file "../../data/magellan/$dataset/contrastive-selective/$size/val.csv" \
          --eval_file "../../data/magellan/$dataset/contrastive-selective/$size/test.csv" \
          --output_dir "../result/valentwin-$dataset-n-$size-hn-10-selective-neg-lr-3e5-bs-512" \
          --num_train_epochs 3 \
          --per_device_train_batch_size 64 \
          --per_device_eval_batch_size 64 \
          --learning_rate 3e-5 \
          --max_seq_length 32 \
          --pooler_type cls \
          --overwrite_output_dir \
          --temp 0.05 \
          --do_train \
          --do_eval \
          --label_names [] \
          --logging_strategy epoch \
          --evaluation_strategy epoch \
          --save_strategy epoch \
          --metric_for_best_model accuracy \
          --load_best_model_at_end \
          --report_to wandb \
          --run_name "valentwin-$dataset-n-$size-hn-10-selective-neg-lr-3e5-bs-512" \
          --fp16 \
          --use_in_batch_instances_as_negatives \
          --restrictive_in_batch_negatives \
          "$@"
    done
done