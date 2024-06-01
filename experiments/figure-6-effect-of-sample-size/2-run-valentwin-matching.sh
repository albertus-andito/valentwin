#!/bin/bash

# ValenTwin
# You can either use the models from the HuggingFace Hub or the models you trained yourself.

declare -a sample_sizes=("100" "200" "300" "400" "500")

declare -a alite_datasets=("25ksep11" "500spend" "1009ipopayments" "amo-ame" "chicago_parks" "cihr" "DCMS_NHM_NHM" "organogram-junior" "school_report" "stockport_contracts")
declare -a magellan_datasets=("academic_papers" "books" "cosmetics" "movies" "restaurants")

# Increasing train and test sizes
for size in "${sample_sizes[@]}"; do
  for dataset in "${alite_datasets[@]}"; do
      python ../../scripts/valentwin-batch-matching.py \
        --pretrained_model_names_or_paths "albertus-andito/valentwin-$dataset-n-$size-hn-10-selective-neg-lr-3e5-bs-512" \
        --measures euc \
        --column_name_weights 0.4 \
        --column_name_measures euc \
        --holistic \
        --tables_root_dir "../../data/alite/$dataset/sample/$size-test" \
        --output_root_dir "./results/$dataset/output/$size" \
        --device cuda:0
  done

  for dataset in "${magellan_datasets[@]}"; do
      python ../../scripts/valentwin-batch-matching.py \
        --pretrained_model_names_or_paths "albertus-andito/valentwin-$dataset-n-$size-hn-10-selective-neg-lr-3e5-bs-512" \
        --measures euc \
        --column_name_weights 0.4 \
        --column_name_measures euc \
        --holistic \
        --tables_root_dir "../../data/magellan/$dataset/sample/$size-test" \
        --output_root_dir "./results/$dataset/output/$size" \
        --device cuda:0
  done
done

# Increasing only test sizes
for size in "${sample_sizes[@]}"; do
  for dataset in "${alite_datasets[@]}"; do
      python ../../scripts/valentwin-batch-matching.py \
        --pretrained_model_names_or_paths "albertus-andito/valentwin-$dataset-n-100-hn-10-selective-neg-lr-3e5-bs-512" \
        --measures euc \
        --column_name_weights 0.4 \
        --column_name_measures euc \
        --holistic \
        --tables_root_dir "../../data/alite/$dataset/sample/$size-test" \
        --output_root_dir "./results/$dataset/output/100-$size" \
        --device cuda:0
  done

  for dataset in "${magellan_datasets[@]}"; do
      python ../../scripts/valentwin-batch-matching.py \
        --pretrained_model_names_or_paths "albertus-andito/valentwin-$dataset-n-100-hn-10-selective-neg-lr-3e5-bs-512" \
        --measures euc \
        --column_name_weights 0.4 \
        --column_name_measures euc \
        --holistic \
        --tables_root_dir "../../data/magellan/$dataset/sample/$size-test" \
        --output_root_dir "./results/$dataset/output/100-$size" \
        --device cuda:0
  done
done

# Increasing only train sizes
for size in "${sample_sizes[@]}"; do
  for dataset in "${alite_datasets[@]}"; do
      python ../../scripts/valentwin-batch-matching.py \
        --pretrained_model_names_or_paths "albertus-andito/valentwin-$dataset-n-$size-hn-10-selective-neg-lr-3e5-bs-512" \
        --measures euc \
        --column_name_weights 0.4 \
        --column_name_measures euc \
        --holistic \
        --tables_root_dir "../../data/alite/$dataset/sample/100-test" \
        --output_root_dir "./results/$dataset/output/$size-100" \
        --device cuda:0
  done

  for dataset in "${magellan_datasets[@]}"; do
      python ../../scripts/valentwin-batch-matching.py \
        --pretrained_model_names_or_paths "albertus-andito/valentwin-$dataset-n-$size-hn-10-selective-neg-lr-3e5-bs-512" \
        --measures euc \
        --column_name_weights 0.4 \
        --column_name_measures euc \
        --holistic \
        --tables_root_dir "../../data/magellan/$dataset/sample/100-test" \
        --output_root_dir "./results/$dataset/output/$size-100" \
        --device cuda:0
  done
done