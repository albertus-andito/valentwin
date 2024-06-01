#!/bin/bash

declare -a alite_datasets=("25ksep11" "500spend" "1009ipopayments" "amo-ame" "chicago_parks" "cihr" "DCMS_NHM_NHM" "organogram-junior" "school_report" "stockport_contracts")
declare -a magellan_datasets=("academic_papers" "books" "cosmetics" "movies" "restaurants")

base_dir="./results"

for dataset in "${alite_datasets[@]}"; do
    python ../../scripts/calculate_metrics.py \
        --input_dir_path "$base_dir/$dataset/output/100" \
        --output_file_path "$base_dir/$dataset/metrics/100.csv" \
        --ground_truth_file_path "../../data/alite/$dataset/ground-truth-mapping/ground-truth.csv" \
        --do_annotate_tp_fp \
        --overwrite_computed_metrics \
        --parallel_workers -1
done

for dataset in "${magellan_datasets[@]}"; do
    python ../../scripts/calculate_metrics.py \
        --input_dir_path "$base_dir/$dataset/output/100" \
        --output_file_path "$base_dir/$dataset/metrics/100.csv" \
        --ground_truth_file_path "../../data/magellan/$dataset/ground-truth-mapping/ground-truth.csv" \
        --do_annotate_tp_fp \
        --overwrite_computed_metrics \
        --parallel_workers -1
done
