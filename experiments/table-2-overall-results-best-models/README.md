# Overall Results Experiments

This folder contains the scripts and configurations to reproduce the overall results experiments in the paper, with the best hyper-parameter settings for each integration set.

0. We assume that you have already sampled the data and generated the contrastive data as described in the `How to Run` section.
1. You can train the models yourself by running the `1-run-valentwin-training.py` script. Otherwise, you can use the models we have trained and shared on HuggingFace.
2. Run the `2-run-valentwin-matching.py` script.
3. Run the `3-run-competitors-matching.py` script. To run Starmie, you need the pre-trained Starmie model, which we have trained and you can download from:
- https://huggingface.co/albertus-andito/starmie-model_swap_cells_constant_column_0
- https://huggingface.co/albertus-andito/starmie-model_shuffle_col_random_column_0
- https://huggingface.co/albertus-andito/starmie-model_sample_row_tfidf_entity_column_0
- https://huggingface.co/albertus-andito/starmie-model_sample_row_ordered_tfidf_entity_column_0
- https://huggingface.co/albertus-andito/starmie-model_sample_row_head_column_0
- https://huggingface.co/albertus-andito/starmie-model_sample_row_frequent_column_0
- https://huggingface.co/albertus-andito/starmie-model_sample_row_alphaHead_column_0
- https://huggingface.co/albertus-andito/starmie-model_replace_cells_alphaHead_row_0
- https://huggingface.co/albertus-andito/starmie-model_drop_num_col_random_column_0
- https://huggingface.co/albertus-andito/starmie-model_drop_nan_col_random_row_0
- https://huggingface.co/albertus-andito/starmie-model_drop_nan_col_random_column_0
- https://huggingface.co/albertus-andito/starmie-model_drop_head_cells_random_column_0
4. Evaluate the matching results using the `4-calculate-metrics` script.