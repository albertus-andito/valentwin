python ../split_and_sample_datasets.py \
    --dataset_dir_paths ../../data/magellan/academic_papers/formatted \
    ../../data/magellan/books/formatted \
    ../../data/magellan/cosmetics/formatted \
    ../../data/magellan/movies/formatted \
    ../../data/magellan/restaurants/formatted \
    ../../data/alite/25ksep11/formatted \
    ../../data/alite/500spend/formatted \
    ../../data/alite/1009ipopayments/formatted \
    ../../data/alite/amo-ame/formatted \
    ../../data/alite/chicago_parks/formatted \
    ../../data/alite/cihr/formatted \
    ../../data/alite/DCMS_NHM_NHM/formatted \
    ../../data/alite/organogram-junior/formatted \
    ../../data/alite/school_report/formatted \
    ../../data/alite/stockport_contracts/formatted \
    --sample_dataset_dir_paths ../../data/magellan/academic_papers/sample \
    ../../data/magellan/books/sample \
    ../../data/magellan/cosmetics/sample \
    ../../data/magellan/movies/sample \
    ../../data/magellan/restaurants/sample \
    ../../data/alite/25ksep11/sample \
    ../../data/alite/500spend/sample \
    ../../data/alite/1009ipopayments/sample \
    ../../data/alite/amo-ame/sample \
    ../../data/alite/chicago_parks/sample \
    ../../data/alite/cihr/sample \
    ../../data/alite/DCMS_NHM_NHM/sample \
    ../../data/alite/organogram-junior/sample \
    ../../data/alite/school_report/sample \
    ../../data/alite/stockport_contracts/sample \
    --sample_sizes 100 200 300 400 500 \
    --split_ratio 0.4 0.2 0.4 \
    --include_all_samples \
    --drop_duplicates \
    --seed 42