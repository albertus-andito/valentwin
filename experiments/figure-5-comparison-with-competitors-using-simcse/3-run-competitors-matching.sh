# ALITE and ALITE-SimCSE
python ../../scripts/alite-batch-matching.py \
    --pretrained_model_names_or_paths bert-base-uncased \
    princeton-nlp/sup-simcse-roberta-base \
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

# Starmie and Starmie-SimCSE (Pre-trained)
python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_sample_row_ordered_tfidf_entity_column_0.pt \
    ../../starmie-models/model_sample_row_ordered_frequent_column_0-simcse.pt \
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

# Starmie and Starmie-SimCSE (Fine-tuned)
python ../../scripts/starmie-batch-matching.py \
    --model_paths ../../starmie-models/model_drop_num_col_random_column_0.pt \
    ../../starmie-models/model_sample_row_ordered_frequent_column_0-simcse.pt \
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
    --train_tables_root_dirs ../../data/alite/25ksep11/sample/100-train \
    ../../data/alite/500spend/sample/100-train \
    ../../data/alite/1009ipopayments/sample/100-train \
    ../../data/alite/amo-ame/sample/100-train \
    ../../data/alite/chicago_parks/sample/100-train \
    ../../data/alite/cihr/sample/100-train \
    ../../data/alite/DCMS_NHM_NHM/sample/100-train \
    ../../data/alite/organogram-junior/sample/100-train \
    ../../data/alite/school_report/sample/100-train \
    ../../data/alite/stockport_contracts/sample/100-train \
    ../../data/magellan/academic_papers/sample/100-train \
    ../../data/magellan/books/sample/100-train \
    ../../data/magellan/cosmetics/sample/100-train \
    ../../data/magellan/movies/sample/100-train \
    ../../data/magellan/restaurants/sample/100-train \
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
    --fine_tune

# DeepJoin
python ../../scripts/deepjoin-batch-matching.py \
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
    --train_tables_root_dirs ../../data/alite/25ksep11/sample/100-train \
    ../../data/alite/500spend/sample/100-train \
    ../../data/alite/1009ipopayments/sample/100-train \
    ../../data/alite/amo-ame/sample/100-train \
    ../../data/alite/chicago_parks/sample/100-train \
    ../../data/alite/cihr/sample/100-train \
    ../../data/alite/DCMS_NHM_NHM/sample/100-train \
    ../../data/alite/organogram-junior/sample/100-train \
    ../../data/alite/school_report/sample/100-train \
    ../../data/alite/stockport_contracts/sample/100-train \
    ../../data/magellan/academic_papers/sample/100-train \
    ../../data/magellan/books/sample/100-train \
    ../../data/magellan/cosmetics/sample/100-train \
    ../../data/magellan/movies/sample/100-train \
    ../../data/magellan/restaurants/sample/100-train \
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
    --train \
    --column_to_text_transformations colname-stat-col \
    --base_model_name sentence-transformers/all-mpnet-base-v2 \
    --shuffle_rates 0.4 \
    --batch_size 16 \
    --num_epochs 10 \
    --warmup_steps 10000

# DeepJoin SimCSE
python ../../scripts/deepjoin-batch-matching.py \
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
    --train_tables_root_dirs ../../data/alite/25ksep11/sample/100-train \
    ../../data/alite/500spend/sample/100-train \
    ../../data/alite/1009ipopayments/sample/100-train \
    ../../data/alite/amo-ame/sample/100-train \
    ../../data/alite/chicago_parks/sample/100-train \
    ../../data/alite/cihr/sample/100-train \
    ../../data/alite/DCMS_NHM_NHM/sample/100-train \
    ../../data/alite/organogram-junior/sample/100-train \
    ../../data/alite/school_report/sample/100-train \
    ../../data/alite/stockport_contracts/sample/100-train \
    ../../data/magellan/academic_papers/sample/100-train \
    ../../data/magellan/books/sample/100-train \
    ../../data/magellan/cosmetics/sample/100-train \
    ../../data/magellan/movies/sample/100-train \
    ../../data/magellan/restaurants/sample/100-train \
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
    --train \
    --column_to_text_transformations colname-col \
    --base_model_name princeton-nlp/sup-simcse-roberta-base \
    --shuffle_rates 0.0 \
    --batch_size 16 \
    --num_epochs 1 \
    --warmup_steps 10000

