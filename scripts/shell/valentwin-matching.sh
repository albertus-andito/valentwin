python ../valentwin-batch-matching.py \
    --pretrained_model_names_or_paths ../result/valentwin-books-n-100-hn-10-selective-neg-lr-3e5-bs-512 \
    --measures cos euc emd \
    --column_name_weights 0.0 0.1 0.2 0.3 0.4 0.5 \
    --column_name_measures cos euc \
    --holistic \
    --tables_root_dir ../../data/magellan/books/sample/100-test \
    --output_root_dir ../../data/magellan/books/output/100 \
    --device cuda:0