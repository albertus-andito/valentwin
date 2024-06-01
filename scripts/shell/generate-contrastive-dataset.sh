python ../create_contrastive_dataset.py \
    --input_dir_paths ../../data/magellan/academic_papers/sample/100-train \
    ../../data/magellan/books/sample/100-train \
    ../../data/magellan/cosmetics/sample/100-train \
    ../../data/magellan/movies/sample/100-train \
    ../../data/magellan/restaurants/sample/100-train \
    ../../data/alite/25ksep11/sample/100-train \
    ../../data/alite/500spend/sample/100-train \
    ../../data/alite/1009ipopayments/sample/100-train \
    ../../data/alite/amo-ame/sample/100-train \
    ../../data/alite/chicago_parks/sample/100-train \
    ../../data/alite/cihr/sample/100-train \
    ../../data/alite/DCMS_NHM_NHM/sample/100-train \
    ../../data/alite/organogram-junior/sample/100-train \
    ../../data/alite/school_report/sample/100-train \
    ../../data/alite/stockport_contracts/sample/100-train \
    ../../data/museum/sample/100-train \
    ../../data/dstl/sample/100-train \
    --output_dir_paths ../../data/magellan/academic_papers/contrastive-selective/100 \
    ../../data/magellan/books/contrastive-selective/100 \
    ../../data/magellan/cosmetics/contrastive-selective/100 \
    ../../data/magellan/movies/contrastive-selective/100 \
    ../../data/magellan/restaurants/contrastive-selective/100 \
    ../../data/alite/25ksep11/contrastive-selective/100 \
    ../../data/alite/500spend/contrastive-selective/100 \
    ../../data/alite/1009ipopayments/contrastive-selective/100 \
    ../../data/alite/amo-ame/contrastive-selective/100 \
    ../../data/alite/chicago_parks/contrastive-selective/100 \
    ../../data/alite/cihr/contrastive-selective/100 \
    ../../data/alite/DCMS_NHM_NHM/contrastive-selective/100 \
    ../../data/alite/organogram-junior/contrastive-selective/100 \
    ../../data/alite/school_report/contrastive-selective/100 \
    ../../data/alite/stockport_contracts/contrastive-selective/100 \
    ../../data/museum/contrastive-selective/100 \
    ../../data/dstl/contrastive-selective/100 \
    --hard_neg_size 10 \
    --with_col_table_names \
    --use_selective_negatives \
    --pretrained_model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --pooling cls \
    --device cuda:0 \
    --num_values_per_item 1

python ../create_contrastive_dataset.py \
    --input_dir_paths ../../data/magellan/academic_papers/sample/200-train \
    ../../data/magellan/books/sample/200-train \
    ../../data/magellan/cosmetics/sample/200-train \
    ../../data/magellan/movies/sample/200-train \
    ../../data/magellan/restaurants/sample/200-train \
    ../../data/alite/25ksep11/sample/200-train \
    ../../data/alite/500spend/sample/200-train \
    ../../data/alite/1009ipopayments/sample/200-train \
    ../../data/alite/amo-ame/sample/200-train \
    ../../data/alite/chicago_parks/sample/200-train \
    ../../data/alite/cihr/sample/200-train \
    ../../data/alite/DCMS_NHM_NHM/sample/200-train \
    ../../data/alite/organogram-junior/sample/200-train \
    ../../data/alite/school_report/sample/200-train \
    ../../data/alite/stockport_contracts/sample/200-train \
    ../../data/museum/sample/200-train \
    ../../data/dstl/sample/200-train \
    --output_dir_paths ../../data/magellan/academic_papers/contrastive-selective/200 \
    ../../data/magellan/books/contrastive-selective/200 \
    ../../data/magellan/cosmetics/contrastive-selective/200 \
    ../../data/magellan/movies/contrastive-selective/200 \
    ../../data/magellan/restaurants/contrastive-selective/200 \
    ../../data/alite/25ksep11/contrastive-selective/200 \
    ../../data/alite/500spend/contrastive-selective/200 \
    ../../data/alite/1009ipopayments/contrastive-selective/200 \
    ../../data/alite/amo-ame/contrastive-selective/200 \
    ../../data/alite/chicago_parks/contrastive-selective/200 \
    ../../data/alite/cihr/contrastive-selective/200 \
    ../../data/alite/DCMS_NHM_NHM/contrastive-selective/200 \
    ../../data/alite/organogram-junior/contrastive-selective/200 \
    ../../data/alite/school_report/contrastive-selective/200 \
    ../../data/alite/stockport_contracts/contrastive-selective/200 \
    ../../data/museum/contrastive-selective/200 \
    ../../data/dstl/contrastive-selective/200 \
    --hard_neg_size 10 \
    --with_col_table_names \
    --use_selective_negatives \
    --pretrained_model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --pooling cls \
    --device cuda:0 \
    --num_values_per_item 1
    
python ../create_contrastive_dataset.py \
    --input_dir_paths ../../data/magellan/academic_papers/sample/300-train \
    ../../data/magellan/books/sample/300-train \
    ../../data/magellan/cosmetics/sample/300-train \
    ../../data/magellan/movies/sample/300-train \
    ../../data/magellan/restaurants/sample/300-train \
    ../../data/alite/25ksep11/sample/300-train \
    ../../data/alite/500spend/sample/300-train \
    ../../data/alite/1009ipopayments/sample/300-train \
    ../../data/alite/amo-ame/sample/300-train \
    ../../data/alite/chicago_parks/sample/300-train \
    ../../data/alite/cihr/sample/300-train \
    ../../data/alite/DCMS_NHM_NHM/sample/300-train \
    ../../data/alite/organogram-junior/sample/300-train \
    ../../data/alite/school_report/sample/300-train \
    ../../data/alite/stockport_contracts/sample/300-train \
    ../../data/museum/sample/300-train \
    ../../data/dstl/sample/300-train \
    --output_dir_paths ../../data/magellan/academic_papers/contrastive-selective/300 \
    ../../data/magellan/books/contrastive-selective/300 \
    ../../data/magellan/cosmetics/contrastive-selective/300 \
    ../../data/magellan/movies/contrastive-selective/300 \
    ../../data/magellan/restaurants/contrastive-selective/300 \
    ../../data/alite/25ksep11/contrastive-selective/300 \
    ../../data/alite/500spend/contrastive-selective/300 \
    ../../data/alite/1009ipopayments/contrastive-selective/300 \
    ../../data/alite/amo-ame/contrastive-selective/300 \
    ../../data/alite/chicago_parks/contrastive-selective/300 \
    ../../data/alite/cihr/contrastive-selective/300 \
    ../../data/alite/DCMS_NHM_NHM/contrastive-selective/300 \
    ../../data/alite/organogram-junior/contrastive-selective/300 \
    ../../data/alite/school_report/contrastive-selective/300 \
    ../../data/alite/stockport_contracts/contrastive-selective/300 \
    ../../data/museum/contrastive-selective/300 \
    ../../data/dstl/contrastive-selective/300 \
    --hard_neg_size 10 \
    --with_col_table_names \
    --use_selective_negatives \
    --pretrained_model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --pooling cls \
    --device cuda:0 \
    --num_values_per_item 1
    
python ../create_contrastive_dataset.py \
    --input_dir_paths ../../data/magellan/academic_papers/sample/400-train \
    ../../data/magellan/books/sample/400-train \
    ../../data/magellan/cosmetics/sample/400-train \
    ../../data/magellan/movies/sample/400-train \
    ../../data/magellan/restaurants/sample/400-train \
    ../../data/alite/25ksep11/sample/400-train \
    ../../data/alite/500spend/sample/400-train \
    ../../data/alite/1009ipopayments/sample/400-train \
    ../../data/alite/amo-ame/sample/400-train \
    ../../data/alite/chicago_parks/sample/400-train \
    ../../data/alite/cihr/sample/400-train \
    ../../data/alite/DCMS_NHM_NHM/sample/400-train \
    ../../data/alite/organogram-junior/sample/400-train \
    ../../data/alite/school_report/sample/400-train \
    ../../data/alite/stockport_contracts/sample/400-train \
    ../../data/museum/sample/400-train \
    ../../data/dstl/sample/400-train \
    --output_dir_paths ../../data/magellan/academic_papers/contrastive-selective/400 \
    ../../data/magellan/books/contrastive-selective/400 \
    ../../data/magellan/cosmetics/contrastive-selective/400 \
    ../../data/magellan/movies/contrastive-selective/400 \
    ../../data/magellan/restaurants/contrastive-selective/400 \
    ../../data/alite/25ksep11/contrastive-selective/400 \
    ../../data/alite/500spend/contrastive-selective/400 \
    ../../data/alite/1009ipopayments/contrastive-selective/400 \
    ../../data/alite/amo-ame/contrastive-selective/400 \
    ../../data/alite/chicago_parks/contrastive-selective/400 \
    ../../data/alite/cihr/contrastive-selective/400 \
    ../../data/alite/DCMS_NHM_NHM/contrastive-selective/400 \
    ../../data/alite/organogram-junior/contrastive-selective/400 \
    ../../data/alite/school_report/contrastive-selective/400 \
    ../../data/alite/stockport_contracts/contrastive-selective/400 \
    ../../data/museum/contrastive-selective/400 \
    ../../data/dstl/contrastive-selective/400 \
    --hard_neg_size 10 \
    --with_col_table_names \
    --use_selective_negatives \
    --pretrained_model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --pooling cls \
    --device cuda:0 \
    --num_values_per_item 1
    
python ../create_contrastive_dataset.py \
    --input_dir_paths ../../data/magellan/academic_papers/sample/500-train \
    ../../data/magellan/books/sample/500-train \
    ../../data/magellan/cosmetics/sample/500-train \
    ../../data/magellan/movies/sample/500-train \
    ../../data/magellan/restaurants/sample/500-train \
    ../../data/alite/25ksep11/sample/500-train \
    ../../data/alite/500spend/sample/500-train \
    ../../data/alite/1009ipopayments/sample/500-train \
    ../../data/alite/amo-ame/sample/500-train \
    ../../data/alite/chicago_parks/sample/500-train \
    ../../data/alite/cihr/sample/500-train \
    ../../data/alite/DCMS_NHM_NHM/sample/500-train \
    ../../data/alite/organogram-junior/sample/500-train \
    ../../data/alite/school_report/sample/500-train \
    ../../data/alite/stockport_contracts/sample/500-train \
    ../../data/museum/sample/500-train \
    ../../data/dstl/sample/500-train \
    --output_dir_paths ../../data/magellan/academic_papers/contrastive-selective/500 \
    ../../data/magellan/books/contrastive-selective/500 \
    ../../data/magellan/cosmetics/contrastive-selective/500 \
    ../../data/magellan/movies/contrastive-selective/500 \
    ../../data/magellan/restaurants/contrastive-selective/500 \
    ../../data/alite/25ksep11/contrastive-selective/500 \
    ../../data/alite/500spend/contrastive-selective/500 \
    ../../data/alite/1009ipopayments/contrastive-selective/500 \
    ../../data/alite/amo-ame/contrastive-selective/500 \
    ../../data/alite/chicago_parks/contrastive-selective/500 \
    ../../data/alite/cihr/contrastive-selective/500 \
    ../../data/alite/DCMS_NHM_NHM/contrastive-selective/500 \
    ../../data/alite/organogram-junior/contrastive-selective/500 \
    ../../data/alite/school_report/contrastive-selective/500 \
    ../../data/alite/stockport_contracts/contrastive-selective/500 \
    ../../data/museum/contrastive-selective/500 \
    ../../data/dstl/contrastive-selective/500 \
    --hard_neg_size 10 \
    --with_col_table_names \
    --use_selective_negatives \
    --pretrained_model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --pooling cls \
    --device cuda:0 \
    --num_values_per_item 1