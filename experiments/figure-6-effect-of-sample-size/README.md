# Increasing Sample Sizes Experiments

This folder contains the scripts and configurations to reproduce the results of increasing both training and test size, train size only, and test size only in the paper.

0. We assume that you have already sampled the data and generated the contrastive data as described in the `How to Run` section.
1. You can train the models yourself by running the `1-run-valentwin-training.py` script. Otherwise, you can use the models we have trained and shared on HuggingFace.
2. Run the `2-run-valentwin-matching.py` script.
3. Run the `3-run-competitors-matching.py` script. To run Starmie, you need the pre-trained Starmie model, which we have trained and you can download from https://huggingface.co/albertus-andito/starmie-model_sample_row_ordered_tfidf_entity_column_0 and https://huggingface.co/albertus-andito/starmie-model_drop_num_col_random_column_0
4. Evaluate the matching results using the `4-calculate-metrics` script.