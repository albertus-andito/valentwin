#!/bin/bash

# ValenTwin
# You can either use the models from the HuggingFace Hub or the models you trained yourself.

python ../../scripts/valentwin-batch-matching.py \
  --pretrained_model_names_or_paths "albertus-andito/valentwin-25ksep11-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10" \
  --measures cos \
  --column_name_weights 0.1 \
  --column_name_measures cos \
  --holistic \
  --tables_root_dir "../../data/alite/25ksep11/sample/100-test" \
  --output_root_dir "./results/25ksep11/output/100" \
  --device cuda:0

python ../../scripts/valentwin-batch-matching.py \
  --pretrained_model_names_or_paths "albertus-andito/valentwin-500spend-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10" \
  --measures cos \
  --column_name_weights 0.0 \
  --column_name_measures cos \
  --holistic \
  --tables_root_dir "../../data/alite/500spend/sample/100-test" \
  --output_root_dir "./results/500spend/output/100" \
  --device cuda:0

python ../../scripts/valentwin-batch-matching.py \
  --pretrained_model_names_or_paths "albertus-andito/valentwin-1009ipopayments-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10" \
  --measures cos \
  --column_name_weights 0.0 \
  --column_name_measures cos \
  --holistic \
  --tables_root_dir "../../data/alite/1009ipopayments/sample/100-test" \
  --output_root_dir "./results/1009ipopayments/output/100" \
  --device cuda:0

python ../../scripts/valentwin-batch-matching.py \
  --pretrained_model_names_or_paths "albertus-andito/valentwin-amo-ame-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10" \
  --measures emd \
  --column_name_weights 0.0 \
  --column_name_measures cos \
  --holistic \
  --tables_root_dir "../../data/alite/amo-ame/sample/100-test" \
  --output_root_dir "./results/amo-ame/output/100" \
  --device cuda:0

python ../../scripts/valentwin-batch-matching.py \
  --pretrained_model_names_or_paths "albertus-andito/valentwin-chicago_parks-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10" \
  --measures emd \
  --column_name_weights 0.5 \
  --column_name_measures euc \
  --holistic \
  --tables_root_dir "../../data/alite/chicago_parks/sample/100-test" \
  --output_root_dir "./results/chicago_parks/output/100" \
  --device cuda:0

python ../../scripts/valentwin-batch-matching.py \
  --pretrained_model_names_or_paths "albertus-andito/valentwin-cihr-n-100-hn-10-selective-neginter-lr-3e5-bs-512" \
  --measures euc \
  --column_name_weights 0.1 \
  --column_name_measures euc \
  --holistic \
  --tables_root_dir "../../data/alite/cihr/sample/100-test" \
  --output_root_dir "./results/cihr/output/100" \
  --device cuda:0

python ../../scripts/valentwin-batch-matching.py \
  --pretrained_model_names_or_paths "albertus-andito/valentwin-DCMS_NHM_NHM-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10" \
  --measures cos \
  --column_name_weights 0.0 \
  --column_name_measures cos \
  --holistic \
  --tables_root_dir "../../data/alite/DCMS_NHM_NHM/sample/100-test" \
  --output_root_dir "./results/DCMS_NHM_NHM/output/100" \
  --device cuda:0

python ../../scripts/valentwin-batch-matching.py \
  --pretrained_model_names_or_paths "albertus-andito/valentwin-organogram-junior-n-100-hn-10-selective-neg-lr-3e5-bs-512-ep-10" \
  --measures emd \
  --column_name_weights 0.4 \
  --column_name_measures euc \
  --holistic \
  --tables_root_dir "../../data/alite/organogram-junior/sample/100-test" \
  --output_root_dir "./results/organogram-junior/output/100" \
  --device cuda:0

python ../../scripts/valentwin-batch-matching.py \
  --pretrained_model_names_or_paths "albertus-andito/valentwin-school_report-n-100-hn-10-selective-neginter-lr-3e5-bs-512-ep-10" \
  --measures cos \
  --column_name_weights 0.6 \
  --column_name_measures euc \
  --holistic \
  --tables_root_dir "../../data/alite/school_report/sample/100-test" \
  --output_root_dir "./results/school_report/output/100" \
  --device cuda:0

python ../../scripts/valentwin-batch-matching.py \
  --pretrained_model_names_or_paths "albertus-andito/valentwin-stockport_contracts-n-100-hn-10-selective-neg-lr-3e5-bs-512" \
  --measures euc \
  --column_name_weights 0.2 \
  --column_name_measures euc \
  --holistic \
  --tables_root_dir "../../data/alite/stockport_contracts/sample/100-test" \
  --output_root_dir "./results/stockport_contracts/output/100" \
  --device cuda:0

python ../../scripts/valentwin-batch-matching.py \
  --pretrained_model_names_or_paths "albertus-andito/valentwin-academic_papers-n-100-hn-10-selective-neg-lr-3e5-bs-512-ep-10" \
  --measures cos \
  --column_name_weights 0.5 \
  --column_name_measures euc \
  --holistic \
  --tables_root_dir "../../data/magellan/academic_papers/sample/100-test" \
  --output_root_dir "./results/academic_papers/output/100" \
  --device cuda:0

python ../../scripts/valentwin-batch-matching.py \
  --pretrained_model_names_or_paths "albertus-andito/valentwin-books-n-100-hn-10-selective-neg-lr-3e5-bs-512" \
  --measures emd \
  --column_name_weights 0.6 \
  --column_name_measures cos \
  --holistic \
  --tables_root_dir "../../data/magellan/books/sample/100-test" \
  --output_root_dir "./results/books/output/100" \
  --device cuda:0

python ../../scripts/valentwin-batch-matching.py \
  --pretrained_model_names_or_paths "albertus-andito/valentwin-cosmetics-n-100-hn-10-selective-neg-lr-3e5-bs-512" \
  --measures emd \
  --column_name_weights 0.0 \
  --column_name_measures cos \
  --holistic \
  --tables_root_dir "../../data/magellan/cosmetics/sample/100-test" \
  --output_root_dir "./results/cosmetics/output/100" \
  --device cuda:0

python ../../scripts/valentwin-batch-matching.py \
  --pretrained_model_names_or_paths "albertus-andito/valentwin-movies-n-100-hn-10-selective-neg-lr-3e5-bs-512-ep-10" \
  --measures emd \
  --column_name_weights 0.0 \
  --column_name_measures cos \
  --holistic \
  --tables_root_dir "../../data/magellan/movies/sample/100-test" \
  --output_root_dir "./results/movies/output/100" \
  --device cuda:0

python ../../scripts/valentwin-batch-matching.py \
  --pretrained_model_names_or_paths "albertus-andito/valentwin-restaurants-n-100-hn-10-selective-neg-lr-3e5-bs-512" \
  --measures emd \
  --column_name_weights 0.0 \
  --column_name_measures cos \
  --holistic \
  --tables_root_dir "../../data/magellan/restaurants/sample/100-test" \
  --output_root_dir "./results/restaurants/output/100" \
  --device cuda:0
