#!/bin/bash

declare -a sample_sizes=("100" "200" "300" "400" "500")

# Increasing training and test sizes
for size in "${sample_sizes[@]}"; do
  # Valentine baselines
  python ../../scripts/valentine-batch-matching.py \
      --algorithms coma_schema_instance \
      --tables_root_dirs "../../data/alite/25ksep11/sample/$size-test" \
      "../../data/alite/500spend/sample/$size-test" \
      "../../data/alite/1009ipopayments/sample/$size-test" \
      "../../data/alite/amo-ame/sample/$size-test" \
      "../../data/alite/chicago_parks/sample/$size-test" \
      "../../data/alite/cihr/sample/$size-test" \
      "../../data/alite/DCMS_NHM_NHM/sample/$size-test" \
      "../../data/alite/organogram-junior/sample/$size-test" \
      "../../data/alite/school_report/sample/$size-test" \
      "../../data/alite/stockport_contracts/sample/$size-test" \
      "../../data/magellan/academic_papers/sample/$size-test" \
      "../../data/magellan/books/sample/$size-test" \
      "../../data/magellan/cosmetics/sample/$size-test" \
      "../../data/magellan/movies/sample/$size-test" \
      "../../data/magellan/restaurants/sample/$size-test" \
      --output_root_dirs "./results/25ksep11/output/$size" \
      "./results/500spend/output/$size" \
      "./results/1009ipopayments/output/$size" \
      "./results/amo-ame/output/$size" \
      "./results/chicago_parks/output/$size" \
      "./results/cihr/output/$size" \
      "./results/DCMS_NHM_NHM/output/$size" \
      "./results/organogram-junior/output/$size" \
      "./results/school_report/output/$size" \
      "./results/stockport_contracts/output/$size" \
      "./results/academic_papers/output/$size" \
      "./results/books/output/$size" \
      "./results/cosmetics/output/$size" \
      "./results/movies/output/$size" \
      "./results/restaurants/output/$size" \
      --parallelize
  
  # Starmie (Pre-trained)
  python ../../scripts/starmie-batch-matching.py \
      --model_paths ../../starmie-models/model_sample_row_ordered_tfidf_entity_column_0.pt \
      --tables_root_dirs "../../data/alite/25ksep11/sample/$size-test" \
      "../../data/alite/500spend/sample/$size-test" \
      "../../data/alite/1009ipopayments/sample/$size-test" \
      "../../data/alite/amo-ame/sample/$size-test" \
      "../../data/alite/chicago_parks/sample/$size-test" \
      "../../data/alite/cihr/sample/$size-test" \
      "../../data/alite/DCMS_NHM_NHM/sample/$size-test" \
      "../../data/alite/organogram-junior/sample/$size-test" \
      "../../data/alite/school_report/sample/$size-test" \
      "../../data/alite/stockport_contracts/sample/$size-test" \
      "../../data/magellan/academic_papers/sample/$size-test" \
      "../../data/magellan/books/sample/$size-test" \
      "../../data/magellan/cosmetics/sample/$size-test" \
      "../../data/magellan/movies/sample/$size-test" \
      "../../data/magellan/restaurants/sample/$size-test" \
      --output_root_dirs "./results/25ksep11/output/$size" \
      "./results/500spend/output/$size" \
      "./results/1009ipopayments/output/$size" \
      "./results/amo-ame/output/$size" \
      "./results/chicago_parks/output/$size" \
      "./results/cihr/output/$size" \
      "./results/DCMS_NHM_NHM/output/$size" \
      "./results/organogram-junior/output/$size" \
      "./results/school_report/output/$size" \
      "./results/stockport_contracts/output/$size" \
      "./results/academic_papers/output/$size" \
      "./results/books/output/$size" \
      "./results/cosmetics/output/$size" \
      "./results/movies/output/$size" \
      "./results/restaurants/output/$size" \
  
  # Starmie (Fine-tuned)
  python ../../scripts/starmie-batch-matching.py \
      --model_paths ../../starmie-models/model_drop_num_col_random_column_0.pt \
      --tables_root_dirs "../../data/alite/25ksep11/sample/$size-test" \
      "../../data/alite/500spend/sample/$size-test" \
      "../../data/alite/1009ipopayments/sample/$size-test" \
      "../../data/alite/amo-ame/sample/$size-test" \
      "../../data/alite/chicago_parks/sample/$size-test" \
      "../../data/alite/cihr/sample/$size-test" \
      "../../data/alite/DCMS_NHM_NHM/sample/$size-test" \
      "../../data/alite/organogram-junior/sample/$size-test" \
      "../../data/alite/school_report/sample/$size-test" \
      "../../data/alite/stockport_contracts/sample/$size-test" \
      "../../data/magellan/academic_papers/sample/$size-test" \
      "../../data/magellan/books/sample/$size-test" \
      "../../data/magellan/cosmetics/sample/$size-test" \
      "../../data/magellan/movies/sample/$size-test" \
      "../../data/magellan/restaurants/sample/$size-test" \
      --train_tables_root_dirs "../../data/alite/25ksep11/sample/$size-train" \
      "../../data/alite/500spend/sample/$size-train" \
      "../../data/alite/1009ipopayments/sample/$size-train" \
      "../../data/alite/amo-ame/sample/$size-train" \
      "../../data/alite/chicago_parks/sample/$size-train" \
      "../../data/alite/cihr/sample/$size-train" \
      "../../data/alite/DCMS_NHM_NHM/sample/$size-train" \
      "../../data/alite/organogram-junior/sample/$size-train" \
      "../../data/alite/school_report/sample/$size-train" \
      "../../data/alite/stockport_contracts/sample/$size-train" \
      "../../data/magellan/academic_papers/sample/$size-train" \
      "../../data/magellan/books/sample/$size-train" \
      "../../data/magellan/cosmetics/sample/$size-train" \
      "../../data/magellan/movies/sample/$size-train" \
      "../../data/magellan/restaurants/sample/$size-train" \
      --output_root_dirs "./results/25ksep11/output/$size" \
      "./results/500spend/output/$size" \
      "./results/1009ipopayments/output/$size" \
      "./results/amo-ame/output/$size" \
      "./results/chicago_parks/output/$size" \
      "./results/cihr/output/$size" \
      "./results/DCMS_NHM_NHM/output/$size" \
      "./results/organogram-junior/output/$size" \
      "./results/school_report/output/$size" \
      "./results/stockport_contracts/output/$size" \
      "./results/academic_papers/output/$size" \
      "./results/books/output/$size" \
      "./results/cosmetics/output/$size" \
      "./results/movies/output/$size" \
      "./results/restaurants/output/$size" \
      --fine_tune
  
  # DeepJoin
  python ../../scripts/deepjoin-batch-matching.py \
      --tables_root_dirs "../../data/alite/25ksep11/sample/$size-test" \
      "../../data/alite/500spend/sample/$size-test" \
      "../../data/alite/1009ipopayments/sample/$size-test" \
      "../../data/alite/amo-ame/sample/$size-test" \
      "../../data/alite/chicago_parks/sample/$size-test" \
      "../../data/alite/cihr/sample/$size-test" \
      "../../data/alite/DCMS_NHM_NHM/sample/$size-test" \
      "../../data/alite/organogram-junior/sample/$size-test" \
      "../../data/alite/school_report/sample/$size-test" \
      "../../data/alite/stockport_contracts/sample/$size-test" \
      "../../data/magellan/academic_papers/sample/$size-test" \
      "../../data/magellan/books/sample/$size-test" \
      "../../data/magellan/cosmetics/sample/$size-test" \
      "../../data/magellan/movies/sample/$size-test" \
      "../../data/magellan/restaurants/sample/$size-test" \
      --train_tables_root_dirs "../../data/alite/25ksep11/sample/$size-train" \
      "../../data/alite/500spend/sample/$size-train" \
      "../../data/alite/1009ipopayments/sample/$size-train" \
      "../../data/alite/amo-ame/sample/$size-train" \
      "../../data/alite/chicago_parks/sample/$size-train" \
      "../../data/alite/cihr/sample/$size-train" \
      "../../data/alite/DCMS_NHM_NHM/sample/$size-train" \
      "../../data/alite/organogram-junior/sample/$size-train" \
      "../../data/alite/school_report/sample/$size-train" \
      "../../data/alite/stockport_contracts/sample/$size-train" \
      "../../data/magellan/academic_papers/sample/$size-train" \
      "../../data/magellan/books/sample/$size-train" \
      "../../data/magellan/cosmetics/sample/$size-train" \
      "../../data/magellan/movies/sample/$size-train" \
      "../../data/magellan/restaurants/sample/$size-train" \
      --output_root_dirs "./results/25ksep11/output/$size" \
      "./results/500spend/output/$size" \
      "./results/1009ipopayments/output/$size" \
      "./results/amo-ame/output/$size" \
      "./results/chicago_parks/output/$size" \
      "./results/cihr/output/$size" \
      "./results/DCMS_NHM_NHM/output/$size" \
      "./results/organogram-junior/output/$size" \
      "./results/school_report/output/$size" \
      "./results/stockport_contracts/output/$size" \
      "./results/academic_papers/output/$size" \
      "./results/books/output/$size" \
      "./results/cosmetics/output/$size" \
      "./results/movies/output/$size" \
      "./results/restaurants/output/$size" \
      --train \
      --column_to_text_transformations colname-stat-col \
      --base_model_name sentence-transformers/all-mpnet-base-v2 \
      --shuffle_rates 0.4 \
      --batch_size 16 \
      --num_epochs 10 \
      --warmup_steps 10000
done

# Increasing only test sizes
for size in "${sample_sizes[@]}"; do
  # Valentine baselines
  python ../../scripts/valentine-batch-matching.py \
      --algorithms coma_schema_instance \
      --tables_root_dirs "../../data/alite/25ksep11/sample/$size-test" \
      "../../data/alite/500spend/sample/$size-test" \
      "../../data/alite/1009ipopayments/sample/$size-test" \
      "../../data/alite/amo-ame/sample/$size-test" \
      "../../data/alite/chicago_parks/sample/$size-test" \
      "../../data/alite/cihr/sample/$size-test" \
      "../../data/alite/DCMS_NHM_NHM/sample/$size-test" \
      "../../data/alite/organogram-junior/sample/$size-test" \
      "../../data/alite/school_report/sample/$size-test" \
      "../../data/alite/stockport_contracts/sample/$size-test" \
      "../../data/magellan/academic_papers/sample/$size-test" \
      "../../data/magellan/books/sample/$size-test" \
      "../../data/magellan/cosmetics/sample/$size-test" \
      "../../data/magellan/movies/sample/$size-test" \
      "../../data/magellan/restaurants/sample/$size-test" \
      --output_root_dirs "./results/25ksep11/output/100-$size" \
      "./results/500spend/output/100-$size" \
      "./results/1009ipopayments/output/100-$size" \
      "./results/amo-ame/output/100-$size" \
      "./results/chicago_parks/output/100-$size" \
      "./results/cihr/output/100-$size" \
      "./results/DCMS_NHM_NHM/output/100-$size" \
      "./results/organogram-junior/output/100-$size" \
      "./results/school_report/output/100-$size" \
      "./results/stockport_contracts/output/100-$size" \
      "./results/academic_papers/output/100-$size" \
      "./results/books/output/100-$size" \
      "./results/cosmetics/output/100-$size" \
      "./results/movies/output/100-$size" \
      "./results/restaurants/output/100-$size" \
      --parallelize
  
  # Starmie (Pre-trained)
  python ../../scripts/starmie-batch-matching.py \
      --model_paths ../../starmie-models/model_sample_row_ordered_tfidf_entity_column_0.pt \
      --tables_root_dirs "../../data/alite/25ksep11/sample/$size-test" \
      "../../data/alite/500spend/sample/$size-test" \
      "../../data/alite/1009ipopayments/sample/$size-test" \
      "../../data/alite/amo-ame/sample/$size-test" \
      "../../data/alite/chicago_parks/sample/$size-test" \
      "../../data/alite/cihr/sample/$size-test" \
      "../../data/alite/DCMS_NHM_NHM/sample/$size-test" \
      "../../data/alite/organogram-junior/sample/$size-test" \
      "../../data/alite/school_report/sample/$size-test" \
      "../../data/alite/stockport_contracts/sample/$size-test" \
      "../../data/magellan/academic_papers/sample/$size-test" \
      "../../data/magellan/books/sample/$size-test" \
      "../../data/magellan/cosmetics/sample/$size-test" \
      "../../data/magellan/movies/sample/$size-test" \
      "../../data/magellan/restaurants/sample/$size-test" \
      --output_root_dirs "./results/25ksep11/output/100-$size" \
      "./results/500spend/output/100-$size" \
      "./results/1009ipopayments/output/100-$size" \
      "./results/amo-ame/output/100-$size" \
      "./results/chicago_parks/output/100-$size" \
      "./results/cihr/output/100-$size" \
      "./results/DCMS_NHM_NHM/output/100-$size" \
      "./results/organogram-junior/output/100-$size" \
      "./results/school_report/output/100-$size" \
      "./results/stockport_contracts/output/100-$size" \
      "./results/academic_papers/output/100-$size" \
      "./results/books/output/100-$size" \
      "./results/cosmetics/output/100-$size" \
      "./results/movies/output/100-$size" \
      "./results/restaurants/output/100-$size" \
  
  # Starmie (Fine-tuned)
  python ../../scripts/starmie-batch-matching.py \
      --model_paths ../../starmie-models/model_drop_num_col_random_column_0.pt \
      --tables_root_dirs "../../data/alite/25ksep11/sample/$size-test" \
      "../../data/alite/500spend/sample/$size-test" \
      "../../data/alite/1009ipopayments/sample/$size-test" \
      "../../data/alite/amo-ame/sample/$size-test" \
      "../../data/alite/chicago_parks/sample/$size-test" \
      "../../data/alite/cihr/sample/$size-test" \
      "../../data/alite/DCMS_NHM_NHM/sample/$size-test" \
      "../../data/alite/organogram-junior/sample/$size-test" \
      "../../data/alite/school_report/sample/$size-test" \
      "../../data/alite/stockport_contracts/sample/$size-test" \
      "../../data/magellan/academic_papers/sample/$size-test" \
      "../../data/magellan/books/sample/$size-test" \
      "../../data/magellan/cosmetics/sample/$size-test" \
      "../../data/magellan/movies/sample/$size-test" \
      "../../data/magellan/restaurants/sample/$size-test" \
      --train_tables_root_dirs "../../data/alite/25ksep11/sample/100-train" \
      "../../data/alite/500spend/sample/100-train" \
      "../../data/alite/1009ipopayments/sample/100-train" \
      "../../data/alite/amo-ame/sample/100-train" \
      "../../data/alite/chicago_parks/sample/100-train" \
      "../../data/alite/cihr/sample/100-train" \
      "../../data/alite/DCMS_NHM_NHM/sample/100-train" \
      "../../data/alite/organogram-junior/sample/100-train" \
      "../../data/alite/school_report/sample/100-train" \
      "../../data/alite/stockport_contracts/sample/100-train" \
      "../../data/magellan/academic_papers/sample/100-train" \
      "../../data/magellan/books/sample/100-train" \
      "../../data/magellan/cosmetics/sample/100-train" \
      "../../data/magellan/movies/sample/100-train" \
      "../../data/magellan/restaurants/sample/100-train" \
      --output_root_dirs "./results/25ksep11/output/100-$size" \
      "./results/500spend/output/100-$size" \
      "./results/1009ipopayments/output/100-$size" \
      "./results/amo-ame/output/100-$size" \
      "./results/chicago_parks/output/100-$size" \
      "./results/cihr/output/100-$size" \
      "./results/DCMS_NHM_NHM/output/100-$size" \
      "./results/organogram-junior/output/100-$size" \
      "./results/school_report/output/100-$size" \
      "./results/stockport_contracts/output/100-$size" \
      "./results/academic_papers/output/100-$size" \
      "./results/books/output/100-$size" \
      "./results/cosmetics/output/100-$size" \
      "./results/movies/output/100-$size" \
      "./results/restaurants/output/100-$size" \
      --fine_tune
  
  # DeepJoin
  python ../../scripts/deepjoin-batch-matching.py \
      --tables_root_dirs "../../data/alite/25ksep11/sample/$size-test" \
      "../../data/alite/500spend/sample/$size-test" \
      "../../data/alite/1009ipopayments/sample/$size-test" \
      "../../data/alite/amo-ame/sample/$size-test" \
      "../../data/alite/chicago_parks/sample/$size-test" \
      "../../data/alite/cihr/sample/$size-test" \
      "../../data/alite/DCMS_NHM_NHM/sample/$size-test" \
      "../../data/alite/organogram-junior/sample/$size-test" \
      "../../data/alite/school_report/sample/$size-test" \
      "../../data/alite/stockport_contracts/sample/$size-test" \
      "../../data/magellan/academic_papers/sample/$size-test" \
      "../../data/magellan/books/sample/$size-test" \
      "../../data/magellan/cosmetics/sample/$size-test" \
      "../../data/magellan/movies/sample/$size-test" \
      "../../data/magellan/restaurants/sample/$size-test" \
      --train_tables_root_dirs "../../data/alite/25ksep11/sample/100-train" \
      "../../data/alite/500spend/sample/100-train" \
      "../../data/alite/1009ipopayments/sample/100-train" \
      "../../data/alite/amo-ame/sample/100-train" \
      "../../data/alite/chicago_parks/sample/100-train" \
      "../../data/alite/cihr/sample/100-train" \
      "../../data/alite/DCMS_NHM_NHM/sample/100-train" \
      "../../data/alite/organogram-junior/sample/100-train" \
      "../../data/alite/school_report/sample/100-train" \
      "../../data/alite/stockport_contracts/sample/100-train" \
      "../../data/magellan/academic_papers/sample/100-train" \
      "../../data/magellan/books/sample/100-train" \
      "../../data/magellan/cosmetics/sample/100-train" \
      "../../data/magellan/movies/sample/100-train" \
      "../../data/magellan/restaurants/sample/100-train" \
      --output_root_dirs "./results/25ksep11/output/100-$size" \
      "./results/500spend/output/100-$size" \
      "./results/1009ipopayments/output/100-$size" \
      "./results/amo-ame/output/100-$size" \
      "./results/chicago_parks/output/100-$size" \
      "./results/cihr/output/100-$size" \
      "./results/DCMS_NHM_NHM/output/100-$size" \
      "./results/organogram-junior/output/100-$size" \
      "./results/school_report/output/100-$size" \
      "./results/stockport_contracts/output/100-$size" \
      "./results/academic_papers/output/100-$size" \
      "./results/books/output/100-$size" \
      "./results/cosmetics/output/100-$size" \
      "./results/movies/output/100-$size" \
      "./results/restaurants/output/100-$size" \
      --train \
      --column_to_text_transformations colname-stat-col \
      --base_model_name sentence-transformers/all-mpnet-base-v2 \
      --shuffle_rates 0.4 \
      --batch_size 16 \
      --num_epochs 10 \
      --warmup_steps 10000
done

# Increasing only training sizes
for size in "${sample_sizes[@]}"; do
  # Valentine baselines
  python ../../scripts/valentine-batch-matching.py \
      --algorithms coma_schema_instance \
      --tables_root_dirs "../../data/alite/25ksep11/sample/100-test" \
      "../../data/alite/500spend/sample/100-test" \
      "../../data/alite/1009ipopayments/sample/100-test" \
      "../../data/alite/amo-ame/sample/100-test" \
      "../../data/alite/chicago_parks/sample/100-test" \
      "../../data/alite/cihr/sample/100-test" \
      "../../data/alite/DCMS_NHM_NHM/sample/100-test" \
      "../../data/alite/organogram-junior/sample/100-test" \
      "../../data/alite/school_report/sample/100-test" \
      "../../data/alite/stockport_contracts/sample/100-test" \
      "../../data/magellan/academic_papers/sample/100-test" \
      "../../data/magellan/books/sample/100-test" \
      "../../data/magellan/cosmetics/sample/100-test" \
      "../../data/magellan/movies/sample/100-test" \
      "../../data/magellan/restaurants/sample/100-test" \
      --output_root_dirs "./results/25ksep11/output/$size-100" \
      "./results/500spend/output/$size-100" \
      "./results/1009ipopayments/output/$size-100" \
      "./results/amo-ame/output/$size-100" \
      "./results/chicago_parks/output/$size-100" \
      "./results/cihr/output/$size-100" \
      "./results/DCMS_NHM_NHM/output/$size-100" \
      "./results/organogram-junior/output/$size-100" \
      "./results/school_report/output/$size-100" \
      "./results/stockport_contracts/output/$size-100" \
      "./results/academic_papers/output/$size-100" \
      "./results/books/output/$size-100" \
      "./results/cosmetics/output/$size-100" \
      "./results/movies/output/$size-100" \
      "./results/restaurants/output/$size-100" \
      --parallelize
  
  # Starmie (Pre-trained)
  python ../../scripts/starmie-batch-matching.py \
      --model_paths ../../starmie-models/model_sample_row_ordered_tfidf_entity_column_0.pt \
      --tables_root_dirs "../../data/alite/25ksep11/sample/100-test" \
      "../../data/alite/500spend/sample/100-test" \
      "../../data/alite/1009ipopayments/sample/100-test" \
      "../../data/alite/amo-ame/sample/100-test" \
      "../../data/alite/chicago_parks/sample/100-test" \
      "../../data/alite/cihr/sample/100-test" \
      "../../data/alite/DCMS_NHM_NHM/sample/100-test" \
      "../../data/alite/organogram-junior/sample/100-test" \
      "../../data/alite/school_report/sample/100-test" \
      "../../data/alite/stockport_contracts/sample/100-test" \
      "../../data/magellan/academic_papers/sample/100-test" \
      "../../data/magellan/books/sample/100-test" \
      "../../data/magellan/cosmetics/sample/100-test" \
      "../../data/magellan/movies/sample/100-test" \
      "../../data/magellan/restaurants/sample/100-test" \
      --output_root_dirs "./results/25ksep11/output/$size-100" \
      "./results/500spend/output/$size-100" \
      "./results/1009ipopayments/output/$size-100" \
      "./results/amo-ame/output/$size-100" \
      "./results/chicago_parks/output/$size-100" \
      "./results/cihr/output/$size-100" \
      "./results/DCMS_NHM_NHM/output/$size-100" \
      "./results/organogram-junior/output/$size-100" \
      "./results/school_report/output/$size-100" \
      "./results/stockport_contracts/output/$size-100" \
      "./results/academic_papers/output/$size-100" \
      "./results/books/output/$size-100" \
      "./results/cosmetics/output/$size-100" \
      "./results/movies/output/$size-100" \
      "./results/restaurants/output/$size-100" \
  
  # Starmie (Fine-tuned)
  python ../../scripts/starmie-batch-matching.py \
      --model_paths ../../starmie-models/model_drop_num_col_random_column_0.pt \
      --tables_root_dirs "../../data/alite/25ksep11/sample/100-test" \
      "../../data/alite/500spend/sample/100-test" \
      "../../data/alite/1009ipopayments/sample/100-test" \
      "../../data/alite/amo-ame/sample/100-test" \
      "../../data/alite/chicago_parks/sample/100-test" \
      "../../data/alite/cihr/sample/100-test" \
      "../../data/alite/DCMS_NHM_NHM/sample/100-test" \
      "../../data/alite/organogram-junior/sample/100-test" \
      "../../data/alite/school_report/sample/100-test" \
      "../../data/alite/stockport_contracts/sample/100-test" \
      "../../data/magellan/academic_papers/sample/100-test" \
      "../../data/magellan/books/sample/100-test" \
      "../../data/magellan/cosmetics/sample/100-test" \
      "../../data/magellan/movies/sample/100-test" \
      "../../data/magellan/restaurants/sample/100-test" \
      --train_tables_root_dirs "../../data/alite/25ksep11/sample/$size-train" \
      "../../data/alite/500spend/sample/$size-train" \
      "../../data/alite/1009ipopayments/sample/$size-train" \
      "../../data/alite/amo-ame/sample/$size-train" \
      "../../data/alite/chicago_parks/sample/$size-train" \
      "../../data/alite/cihr/sample/$size-train" \
      "../../data/alite/DCMS_NHM_NHM/sample/$size-train" \
      "../../data/alite/organogram-junior/sample/$size-train" \
      "../../data/alite/school_report/sample/$size-train" \
      "../../data/alite/stockport_contracts/sample/$size-train" \
      "../../data/magellan/academic_papers/sample/$size-train" \
      "../../data/magellan/books/sample/$size-train" \
      "../../data/magellan/cosmetics/sample/$size-train" \
      "../../data/magellan/movies/sample/$size-train" \
      "../../data/magellan/restaurants/sample/$size-train" \
      --output_root_dirs "./results/25ksep11/output/$size-100" \
      "./results/500spend/output/$size-100" \
      "./results/1009ipopayments/output/$size-100" \
      "./results/amo-ame/output/$size-100" \
      "./results/chicago_parks/output/$size-100" \
      "./results/cihr/output/$size-100" \
      "./results/DCMS_NHM_NHM/output/$size-100" \
      "./results/organogram-junior/output/$size-100" \
      "./results/school_report/output/$size-100" \
      "./results/stockport_contracts/output/$size-100" \
      "./results/academic_papers/output/$size-100" \
      "./results/books/output/$size-100" \
      "./results/cosmetics/output/$size-100" \
      "./results/movies/output/$size-100" \
      "./results/restaurants/output/$size-100" \
      --fine_tune
  
  # DeepJoin
  python ../../scripts/deepjoin-batch-matching.py \
      --tables_root_dirs "../../data/alite/25ksep11/sample/100-test" \
      "../../data/alite/500spend/sample/100-test" \
      "../../data/alite/1009ipopayments/sample/100-test" \
      "../../data/alite/amo-ame/sample/100-test" \
      "../../data/alite/chicago_parks/sample/100-test" \
      "../../data/alite/cihr/sample/100-test" \
      "../../data/alite/DCMS_NHM_NHM/sample/100-test" \
      "../../data/alite/organogram-junior/sample/100-test" \
      "../../data/alite/school_report/sample/100-test" \
      "../../data/alite/stockport_contracts/sample/100-test" \
      "../../data/magellan/academic_papers/sample/100-test" \
      "../../data/magellan/books/sample/100-test" \
      "../../data/magellan/cosmetics/sample/100-test" \
      "../../data/magellan/movies/sample/100-test" \
      "../../data/magellan/restaurants/sample/100-test" \
      --train_tables_root_dirs "../../data/alite/25ksep11/sample/$size-train" \
      "../../data/alite/500spend/sample/$size-train" \
      "../../data/alite/1009ipopayments/sample/$size-train" \
      "../../data/alite/amo-ame/sample/$size-train" \
      "../../data/alite/chicago_parks/sample/$size-train" \
      "../../data/alite/cihr/sample/$size-train" \
      "../../data/alite/DCMS_NHM_NHM/sample/$size-train" \
      "../../data/alite/organogram-junior/sample/$size-train" \
      "../../data/alite/school_report/sample/$size-train" \
      "../../data/alite/stockport_contracts/sample/$size-train" \
      "../../data/magellan/academic_papers/sample/$size-train" \
      "../../data/magellan/books/sample/$size-train" \
      "../../data/magellan/cosmetics/sample/$size-train" \
      "../../data/magellan/movies/sample/$size-train" \
      "../../data/magellan/restaurants/sample/$size-train" \
      --output_root_dirs "./results/25ksep11/output/$size-100" \
      "./results/500spend/output/$size-100" \
      "./results/1009ipopayments/output/$size-100" \
      "./results/amo-ame/output/$size-100" \
      "./results/chicago_parks/output/$size-100" \
      "./results/cihr/output/$size-100" \
      "./results/DCMS_NHM_NHM/output/$size-100" \
      "./results/organogram-junior/output/$size-100" \
      "./results/school_report/output/$size-100" \
      "./results/stockport_contracts/output/$size-100" \
      "./results/academic_papers/output/$size-100" \
      "./results/books/output/$size-100" \
      "./results/cosmetics/output/$size-100" \
      "./results/movies/output/$size-100" \
      "./results/restaurants/output/$size-100" \
      --train \
      --column_to_text_transformations colname-stat-col \
      --base_model_name sentence-transformers/all-mpnet-base-v2 \
      --shuffle_rates 0.4 \
      --batch_size 16 \
      --num_epochs 10 \
      --warmup_steps 10000
done
