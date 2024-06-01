#!/bin/bash

declare -a training_sizes=("100" "200" "300" "400" "500")

declare -a alite_datasets=("25ksep11" "500spend" "1009ipopayments" "amo-ame" "chicago_parks" "cihr" "DCMS_NHM_NHM" "organogram-junior" "school_report" "stockport_contracts")
declare -a magellan_datasets=("academic_papers" "books" "cosmetics" "movies" "restaurants")

base_dir="./results"

# Increasing train and test sizes
for size in "${training_sizes[@]}"; do
  for dataset in "${alite_datasets[@]}"; do
      python ../../scripts/calculate_metrics.py \
          --input_dir_path "$base_dir/$dataset/output/$size" \
          --output_file_path "$base_dir/$dataset/metrics/$size.csv" \
          --ground_truth_file_path "../../data/alite/$dataset/ground-truth-mapping/ground-truth.csv" \
          --do_annotate_tp_fp \
          --overwrite_computed_metrics \
          --parallel_workers -1
  done

  for dataset in "${magellan_datasets[@]}"; do
      python ../../scripts/calculate_metrics.py \
          --input_dir_path "$base_dir/$dataset/output/$size" \
          --output_file_path "$base_dir/$dataset/metrics/$size.csv" \
          --ground_truth_file_path "../../data/magellan/$dataset/ground-truth-mapping/ground-truth.csv" \
          --do_annotate_tp_fp \
          --overwrite_computed_metrics \
          --parallel_workers -1
  done
done

# Increasing only test sizes
for size in "${training_sizes[@]}"; do
  for dataset in "${alite_datasets[@]}"; do
      python ../../scripts/calculate_metrics.py \
          --input_dir_path "$base_dir/$dataset/output/100-$size" \
          --output_file_path "$base_dir/$dataset/metrics/100-$size.csv" \
          --ground_truth_file_path "../../data/alite/$dataset/ground-truth-mapping/ground-truth.csv" \
          --do_annotate_tp_fp \
          --overwrite_computed_metrics \
          --parallel_workers -1
  done

  for dataset in "${magellan_datasets[@]}"; do
      python ../../scripts/calculate_metrics.py \
          --input_dir_path "$base_dir/$dataset/output/100-$size" \
          --output_file_path "$base_dir/$dataset/metrics/100-$size.csv" \
          --ground_truth_file_path "../../data/magellan/$dataset/ground-truth-mapping/ground-truth.csv" \
          --do_annotate_tp_fp \
          --overwrite_computed_metrics \
          --parallel_workers -1
  done
done

# Increasing only train sizes
for size in "${training_sizes[@]}"; do
  for dataset in "${alite_datasets[@]}"; do
      python ../../scripts/calculate_metrics.py \
          --input_dir_path "$base_dir/$dataset/output/$size-100" \
          --output_file_path "$base_dir/$dataset/metrics/$size-100.csv" \
          --ground_truth_file_path "../../data/alite/$dataset/ground-truth-mapping/ground-truth.csv" \
          --do_annotate_tp_fp \
          --overwrite_computed_metrics \
          --parallel_workers -1
  done

  for dataset in "${magellan_datasets[@]}"; do
      python ../../scripts/calculate_metrics.py \
          --input_dir_path "$base_dir/$dataset/output/$size-100" \
          --output_file_path "$base_dir/$dataset/metrics/$size-100.csv" \
          --ground_truth_file_path "../../data/magellan/$dataset/ground-truth-mapping/ground-truth.csv" \
          --do_annotate_tp_fp \
          --overwrite_computed_metrics \
          --parallel_workers -1
  done
done