# Valentine baselines
python ../../scripts/valentine-batch-matching.py \
    --algorithms jaccard distribution_based similarity_flooding coma_schema coma_schema_instance cupid \
    --tables_root_dirs ../../data/alite/25ksep11/sample/100-test \
    ../../data/alite/500spend/sample/100-test \
    ../../data/alite/1009ipopayments/sample/100-test \
    ../../data/alite/amo-ame/sample/100-test \
    ../../data/alite/chicago_parks/sample/100-test \
    ../../data/alite/cihr/sample/100-test \
    ../../data/alite/DCMS_NHM_NHM/sample/100-test \
    ../../data/alite/organogram-junior/sample/100-test \
    ../../data/alite/school_report/sample/100-test \
    ../../data/alite/stockport_contracts/sample/100-test \
    ../../data/magellan/academic_papers/sample/100-test \
    ../../data/magellan/books/sample/100-test \
    ../../data/magellan/cosmetics/sample/100-test \
    ../../data/magellan/movies/sample/100-test \
    ../../data/magellan/restaurants/sample/100-test \
    --output_root_dirs ./results/25ksep11/output/100 \
    ./results/500spend/output/100 \
    ./results/1009ipopayments/output/100 \
    ./results/amo-ame/output/100 \
    ./results/chicago_parks/output/100 \
    ./results/cihr/output/100 \
    ./results/DCMS_NHM_NHM/output/100 \
    ./results/organogram-junior/output/100 \
    ./results/school_report/output/100 \
    ./results/stockport_contracts/output/100 \
    ./results/academic_papers/output/100 \
    ./results/books/output/100 \
    ./results/cosmetics/output/100 \
    ./results/movies/output/100 \
    ./results/restaurants/output/100 \
    --parallelize

# ALITE
python ../../scripts/alite-batch-matching.py \
    --pretrained_model_names_or_paths bert-base-uncased \
    --tables_root_dirs ../../data/alite/25ksep11/sample/100-test \
    ../../data/alite/500spend/sample/100-test \
    ../../data/alite/1009ipopayments/sample/100-test \
    ../../data/alite/amo-ame/sample/100-test \
    ../../data/alite/chicago_parks/sample/100-test \
    ../../data/alite/cihr/sample/100-test \
    ../../data/alite/DCMS_NHM_NHM/sample/100-test \
    ../../data/alite/organogram-junior/sample/100-test \
    ../../data/alite/school_report/sample/100-test \
    ../../data/alite/stockport_contracts/sample/100-test \
    ../../data/magellan/academic_papers/sample/100-test \
    ../../data/magellan/books/sample/100-test \
    ../../data/magellan/cosmetics/sample/100-test \
    ../../data/magellan/movies/sample/100-test \
    ../../data/magellan/restaurants/sample/100-test \
    --output_root_dirs ./results/25ksep11/output/100 \
    ./results/500spend/output/100 \
    ./results/1009ipopayments/output/100 \
    ./results/amo-ame/output/100 \
    ./results/chicago_parks/output/100 \
    ./results/cihr/output/100 \
    ./results/DCMS_NHM_NHM/output/100 \
    ./results/organogram-junior/output/100 \
    ./results/school_report/output/100 \
    ./results/stockport_contracts/output/100 \
    ./results/academic_papers/output/100 \
    ./results/books/output/100 \
    ./results/cosmetics/output/100 \
    ./results/movies/output/100 \
    ./results/restaurants/output/100 \
    --clustering_distance_metric l2 \
    --device cuda:0

# Starmie (Pre-trained)
python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_nan_col_random_column_0.pt \
    --tables_root_dirs ../../data/alite/25ksep11/sample/100-test \
    --output_root_dirs ./results/25ksep11/output/100

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_num_col_random_column_0.pt \
    --tables_root_dirs ../../data/alite/500spend/sample/100-test \
    --output_root_dirs ./results/500spend/output/100

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_num_col_random_column_0.pt \
    --tables_root_dirs ../../data/alite/1009ipopayments/sample/100-test \
    --output_root_dirs ./results/1009ipopayments/output/100

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_num_col_random_column_0.pt \
    --tables_root_dirs ../../data/alite/amo-ame/sample/100-test \
    --output_root_dirs ./results/amo-ame/output/100

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_num_col_random_column_0.pt \
    --tables_root_dirs ../../data/alite/chicago_parks/sample/100-test \
    --output_root_dirs ./results/chicago_parks/output/100

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_swap_cells_constant_column_0.pt \
    --tables_root_dirs ../../data/alite/cihr/sample/100-test \
    --output_root_dirs ./results/cihr/output/100

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_sample_row_tfidf_entity_column_0.pt \
    --tables_root_dirs ../../data/alite/DCMS_NHM_NHM/sample/100-test \
    --output_root_dirs ./results/DCMS_NHM_NHM/output/100

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_head_cells_random_column_0.pt \
    --tables_root_dirs ../../data/alite/organogram-junior/sample/100-test \
    --output_root_dirs ./results/organogram-junior/output/100

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_sample_row_alphaHead_column_0.pt \
    --tables_root_dirs ../../data/alite/school_report/sample/100-test \
    --output_root_dirs ./results/school_report/output/100

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_sample_row_head_column_0.pt \
    --tables_root_dirs ../../data/alite/stockport_contracts/sample/100-test \
    --output_root_dirs ./results/stockport_contracts/output/100

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_nan_col_random_column_0.pt \
    --tables_root_dirs ../../data/magellan/academic_papers/sample/100-test \
    --output_root_dirs ./results/academic_papers/output/100

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_nan_col_random_column_0.pt \
    --tables_root_dirs ../../data/magellan/books/sample/100-test \
    --output_root_dirs ./results/books/output/100

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_shuffle_col_random_column_0.pt \
    --tables_root_dirs ../../data/magellan/cosmetics/sample/100-test \
    --output_root_dirs ./results/cosmetics/output/100

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_nan_col_random_column_0.pt \
    --tables_root_dirs ../../data/magellan/movies/sample/100-test \
    --output_root_dirs ./results/movies/output/100

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_nan_col_random_column_0.pt \
    --tables_root_dirs ../../data/magellan/restaurants/sample/100-test \
    --output_root_dirs ./results/restaurants/output/100


# Starmie (Fine-tuned)
python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_nan_col_random_column_0.pt \
    --tables_root_dirs ../../data/alite/25ksep11/sample/100-test \
    --train_tables_root_dirs ../../data/alite/25ksep11/sample/100-train \
    --output_root_dirs ./results/25ksep11/output/100 \
    --fine_tune

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_replace_cells_alphaHead_row_0.pt \
    --tables_root_dirs ../../data/alite/500spend/sample/100-test \
    --train_tables_root_dirs ../../data/alite/500spend/sample/100-train \
    --output_root_dirs ./results/500spend/output/100 \
    --fine_tune

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_shuffle_col_random_column_0.pt \
    --tables_root_dirs ../../data/alite/1009ipopayments/sample/100-test \
    --train_tables_root_dirs ../../data/alite/1009ipopayments/sample/100-train \
    --output_root_dirs ./results/1009ipopayments/output/100 \
    --fine_tune

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_num_col_random_column_0.pt \
    --tables_root_dirs ../../data/alite/amo-ame/sample/100-test \
    --train_tables_root_dirs ../../data/alite/amo-ame/sample/100-train \
    --output_root_dirs ./results/amo-ame/output/100 \
    --fine_tune

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_num_col_random_column_0.pt \
    --tables_root_dirs ../../data/alite/chicago_parks/sample/100-test \
    --train_tables_root_dirs ../../data/alite/chicago_parks/sample/100-train \
    --output_root_dirs ./results/chicago_parks/output/100 \
    --fine_tune

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_swap_cells_constant_column_0.pt \
    --tables_root_dirs ../../data/alite/cihr/sample/100-test \
    --train_tables_root_dirs ../../data/alite/cihr/sample/100-train \
    --output_root_dirs ./results/cihr/output/100 \
    --fine_tune

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_sample_row_tfidf_entity_column_0.pt \
    --tables_root_dirs ../../data/alite/DCMS_NHM_NHM/sample/100-test \
    --train_tables_root_dirs ../../data/alite/DCMS_NHM_NHM/sample/100-train \
    --output_root_dirs ./results/DCMS_NHM_NHM/output/100 \
    --fine_tune

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_head_cells_random_column_0.pt \
    --tables_root_dirs ../../data/alite/organogram-junior/sample/100-test \
    --train_tables_root_dirs ../../data/alite/organogram-junior/sample/100-train \
    --output_root_dirs ./results/organogram-junior/output/100 \
    --fine_tune

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_sample_row_alphaHead_column_0.pt \
    --tables_root_dirs ../../data/alite/school_report/sample/100-test \
    --train_tables_root_dirs ../../data/alite/school_report/sample/100-train \
    --output_root_dirs ./results/school_report/output/100 \
    --fine_tune

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_sample_row_head_column_0.pt \
    --tables_root_dirs ../../data/alite/stockport_contracts/sample/100-test \
    --train_tables_root_dirs ../../data/alite/stockport_contracts/sample/100-train \
    --output_root_dirs ./results/stockport_contracts/output/100 \
    --fine_tune

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_sample_row_frequent_column_0.pt \
    --tables_root_dirs ../../data/magellan/academic_papers/sample/100-test \
    --train_tables_root_dirs ../../data/magellan/academic_papers/sample/100-train \
    --output_root_dirs ./results/academic_papers/output/100 \
    --fine_tune

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_nan_col_random_column_0.pt \
    --tables_root_dirs ../../data/magellan/books/sample/100-test \
    --train_tables_root_dirs ../../data/magellan/books/sample/100-train \
    --output_root_dirs ./results/books/output/100 \
    --fine_tune

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_nan_col_random_row_0.pt \
    --tables_root_dirs ../../data/magellan/cosmetics/sample/100-test \
    --train_tables_root_dirs ../../data/magellan/cosmetics/sample/100-train \
    --output_root_dirs ./results/cosmetics/output/100 \
    --fine_tune

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_num_col_random_column_0.pt \
    --tables_root_dirs ../../data/magellan/movies/sample/100-test \
    --train_tables_root_dirs ../../data/magellan/movies/sample/100-train \
    --output_root_dirs ./results/movies/output/100 \
    --fine_tune

python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_nan_col_random_column_0.pt \
    --tables_root_dirs ../../data/magellan/restaurants/sample/100-test \
    --train_tables_root_dirs ../../data/magellan/restaurants/sample/100-train \
    --output_root_dirs ./results/restaurants/output/100 \
    --fine_tune

# DeepJoin
python ../../scripts/deepjoin-batch-matching.py \
    --tables_root_dirs ../../data/alite/25ksep11/sample/100-test \
    --train_tables_root_dirs ../../data/alite/25ksep11/sample/100-train \
    --output_root_dirs ./results/25ksep11/output/100 \
    --train \
    --column_to_text_transformations title-colname-stat-col \
    --base_model_name sentence-transformers/all-mpnet-base-v2 \
    --shuffle_rates 0.1 \
    --batch_size 16 \
    --num_epochs 1 \
    --warmup_steps 10000

python ../../scripts/deepjoin-batch-matching.py \
    --tables_root_dirs ../../data/alite/500spend/sample/100-test \
    --train_tables_root_dirs ../../data/alite/500spend/sample/100-train \
    --output_root_dirs ./results/500spend/output/100 \
    --train \
    --column_to_text_transformations title-colname-col \
    --base_model_name sentence-transformers/all-mpnet-base-v2 \
    --shuffle_rates 0.5 \
    --batch_size 16 \
    --num_epochs 1

python ../../scripts/deepjoin-batch-matching.py \
    --tables_root_dirs ../../data/alite/1009ipopayments/sample/100-test \
    --train_tables_root_dirs ../../data/alite/1009ipopayments/sample/100-train \
    --output_root_dirs ./results/1009ipopayments/output/100 \
    --train \
    --column_to_text_transformations col \
    --base_model_name sentence-transformers/all-mpnet-base-v2 \
    --shuffle_rates 0.0 \
    --batch_size 16 \
    --num_epochs 1

python ../../scripts/deepjoin-batch-matching.py \
    --tables_root_dirs ../../data/alite/amo-ame/sample/100-test \
    --train_tables_root_dirs ../../data/alite/amo-ame/sample/100-train \
    --output_root_dirs ./results/amo-ame/output/100 \
    --train \
    --column_to_text_transformations col \
    --base_model_name sentence-transformers/all-mpnet-base-v2 \
    --shuffle_rates 0.3 \
    --batch_size 16 \
    --num_epochs 10

python ../../scripts/deepjoin-batch-matching.py \
    --tables_root_dirs ../../data/alite/chicago_parks/sample/100-test \
    --train_tables_root_dirs ../../data/alite/chicago_parks/sample/100-train \
    --output_root_dirs ./results/chicago_parks/output/100 \
    --train \
    --column_to_text_transformations col \
    --base_model_name sentence-transformers/all-mpnet-base-v2 \
    --shuffle_rates 0.5 \
    --batch_size 16 \
    --num_epochs 1

python ../../scripts/deepjoin-batch-matching.py \
    --tables_root_dirs ../../data/alite/cihr/sample/100-test \
    --train_tables_root_dirs ../../data/alite/cihr/sample/100-train \
    --output_root_dirs ./results/cihr/output/100 \
    --train \
    --column_to_text_transformations col \
    --base_model_name sentence-transformers/all-mpnet-base-v2 \
    --shuffle_rates 0.3 \
    --batch_size 16 \
    --num_epochs 1 \
    --warmup_steps 10000

python ../../scripts/deepjoin-batch-matching.py \
    --tables_root_dirs ../../data/alite/DCMS_NHM_NHM/sample/100-test \
    --train_tables_root_dirs ../../data/alite/DCMS_NHM_NHM/sample/100-train \
    --output_root_dirs ./results/DCMS_NHM_NHM/output/100 \
    --train \
    --column_to_text_transformations col \
    --base_model_name sentence-transformers/all-mpnet-base-v2 \
    --shuffle_rates 0.1 \
    --batch_size 16 \
    --num_epochs 10

python ../../scripts/deepjoin-batch-matching.py \
    --tables_root_dirs ../../data/alite/organogram-junior/sample/100-test \
    --train_tables_root_dirs ../../data/alite/organogram-junior/sample/100-train \
    --output_root_dirs ./results/organogram-junior/output/100 \
    --train \
    --column_to_text_transformations colname-stat-col \
    --base_model_name sentence-transformers/all-mpnet-base-v2 \
    --shuffle_rates 0.1 \
    --batch_size 16 \
    --num_epochs 10 \
    --warmup_steps 10000

python ../../scripts/deepjoin-batch-matching.py \
    --tables_root_dirs ../../data/alite/school_report/sample/100-test \
    --train_tables_root_dirs ../../data/alite/school_report/sample/100-train \
    --output_root_dirs ./results/school_report/output/100 \
    --train \
    --column_to_text_transformations colname-col \
    --base_model_name sentence-transformers/all-mpnet-base-v2 \
    --shuffle_rates 0.1 \
    --batch_size 16 \
    --num_epochs 1 \
    --warmup_steps 10000

python ../../scripts/deepjoin-batch-matching.py \
    --tables_root_dirs ../../data/alite/stockport_contracts/sample/100-test \
    --train_tables_root_dirs ../../data/alite/stockport_contracts/sample/100-train \
    --output_root_dirs ./results/stockport_contracts/output/100 \
    --train \
    --column_to_text_transformations title-colname-col \
    --base_model_name sentence-transformers/all-mpnet-base-v2 \
    --shuffle_rates 0.6 \
    --batch_size 16 \
    --num_epochs 10 \
    --warmup_steps 10000

python ../../scripts/deepjoin-batch-matching.py \
    --tables_root_dirs ../../data/magellan/academic_papers/sample/100-test \
    --train_tables_root_dirs ../../data/magellan/academic_papers/sample/100-train \
    --output_root_dirs ./results/academic_papers/output/100 \
    --train \
    --column_to_text_transformations colname-stat-col \
    --base_model_name sentence-transformers/all-mpnet-base-v2 \
    --shuffle_rates 0.0 \
    --batch_size 16 \
    --num_epochs 10

python ../../scripts/deepjoin-batch-matching.py \
    --tables_root_dirs ../../data/magellan/books/sample/100-test \
    --train_tables_root_dirs ../../data/magellan/books/sample/100-train \
    --output_root_dirs ./results/books/output/100 \
    --train \
    --column_to_text_transformations title-colname-stat-col \
    --base_model_name sentence-transformers/all-mpnet-base-v2 \
    --shuffle_rates 0.3 \
    --batch_size 16 \
    --num_epochs 1

python ../../scripts/deepjoin-batch-matching.py \
    --tables_root_dirs ../../data/magellan/cosmetics/sample/100-test \
    --train_tables_root_dirs ../../data/magellan/cosmetics/sample/100-train \
    --output_root_dirs ./results/cosmetics/output/100 \
    --train \
    --column_to_text_transformations colname-col \
    --base_model_name sentence-transformers/all-mpnet-base-v2 \
    --shuffle_rates 0.2 \
    --batch_size 16 \
    --num_epochs 10

python ../../scripts/deepjoin-batch-matching.py \
    --tables_root_dirs ../../data/magellan/movies/sample/100-test \
    --train_tables_root_dirs ../../data/magellan/movies/sample/100-train \
    --output_root_dirs ./results/movies/output/100 \
    --train \
    --column_to_text_transformations title-colname-stat-col \
    --base_model_name sentence-transformers/all-mpnet-base-v2 \
    --shuffle_rates 0.2 \
    --batch_size 16 \
    --num_epochs 1

python ../../scripts/deepjoin-batch-matching.py \
    --tables_root_dirs ../../data/magellan/restaurants/sample/100-test \
    --train_tables_root_dirs ../../data/magellan/restaurants/sample/100-train \
    --output_root_dirs ./results/restaurants/output/100 \
    --train \
    --column_to_text_transformations colname-stat-col \
    --base_model_name sentence-transformers/all-mpnet-base-v2 \
    --shuffle_rates 0.0 \
    --batch_size 16 \
    --num_epochs 1
