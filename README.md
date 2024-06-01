# ValenTwin: Using Self-Supervised Contrastive Learning on Language Model for Schema Matching
ValenTwin is a schema matching framework that uses self-supervised contrastive learning to train the model, 
uses the model to generate embeddings of table columns, then uses different similarity measures to match the column embeddings.

## Getting Started
1. Clone the repository
```shell
git clone https://github.com/albertus-andito/valentwin.git
cd valentwin
```
2. Install the required packages. It is recommended to use a virtual environment.
```shell
pip install -r requirements.txt
pip install -e .
```

## Datasets
The datasets can be downloaded from https://zenodo.org/records/11413479

We provide two types of zip files for the datasets:
1. `data.zip` contains the raw data files, the ground truth files, the sampled data (n=[100, 200, 300, 400, 500]) used in the experiments, as well as the contrastive data used to train the model.
2. `data-raw.zip` contains only the raw data files and the ground truth files. You can sample the data and generate the contrastive dataset yourself by following step 1 and 2 in the `How to Run` section. 
Download and unzip one of the zip files to the `data` folder.

## Trained Models
<details>
  <summary>Click me</summary>

| Integration Set     | Best Model on Average                                                                                                                                                                 | Best Model on Each Integration Set                                                                                                                                                             |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 25ksep11            | [valentwin-25ksep11-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-25ksep11-n-100-hn-10-selective-neg-lr-3e5-bs-512)                       | [valentwin-25ksep11-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10](https://huggingface.co/albertus-andito/valentwin-25ksep11-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10)                |
| 500spend            | [valentwin-500spend-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-500spend-n-100-hn-10-selective-neg-lr-3e5-bs-512)                       | [valentwin-500spend-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10](https://huggingface.co/albertus-andito/valentwin-500spend-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10)                |
| 1009ipopayments     | [valentwin-1009ipopayments-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-1009ipopayments-n-100-hn-10-selective-neg-lr-3e5-bs-512)         | [valentwin-1009ipopayments-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10](https://huggingface.co/albertus-andito/valentwin-1009ipopayments-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10)  |
| amo-ame             | [valentwin-amo-ame-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-amo-ame-n-100-hn-10-selective-neg-lr-3e5-bs-512)                         | [valentwin-amo-ame-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10](https://huggingface.co/albertus-andito/valentwin-amo-ame-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10)                   |
| chicago_parks       | [valentwin-chicago_parks-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-chicago_parks-n-100-hn-10-selective-neg-lr-3e5-bs-512)             | [valentwin-chicago_parks-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10](https://huggingface.co/albertus-andito/valentwin-chicago_parks-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10)       |
| cihr                | [valentwin-cihr-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-cihr-n-100-hn-10-selective-neg-lr-3e5-bs-512)                               | [valentwin-cihr-n-100-hn-10-selective-neginter-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-cihr-n-100-hn-10-selective-neginter-lr-3e5-bs-512)                               |
| DCMS_NHM_NHM        | [valentwin-DCMS_NHM_NHM-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-DCMS_NHM_NHM-n-100-hn-10-selective-neg-lr-3e5-bs-512)               | [valentwin-DCMS_NHM_NHM-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10](https://huggingface.co/albertus-andito/valentwin-DCMS_NHM_NHM-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10)         |
| organogram-junior   | [valentwin-organogram-junior-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-organogram-junior-n-100-hn-10-selective-neg-lr-3e5-bs-512)     | [valentwin-organogram-junior-n-100-hn-10-selective-neg-lr-3e5-bs-512-ep-10](https://huggingface.co/albertus-andito/valentwin-organogram-junior-n-100-hn-10-selective-neg-lr-3e5-bs-512-ep-10)   |
| school_report       | [valentwin-school_report-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-school_report-n-100-hn-10-selective-neg-lr-3e5-bs-512)             | [valentwin-school_report-n-100-hn-10-selective-neginter-lr-3e5-bs-512-ep-10](https://huggingface.co/albertus-andito/valentwin-school_report-n-100-hn-10-selective-neginter-lr-3e5-bs-512-ep-10) |
| stockport_contracts | [valentwin-stockport_contracts-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-stockport_contracts-n-100-hn-10-selective-neg-lr-3e5-bs-512) | [valentwin-stockport_contracts-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-stockport_contracts-n-100-hn-10-selective-neg-lr-3e5-bs-512)           |
| academic_papers     | [valentwin-academic_papers-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-academic_papers-n-100-hn-10-selective-neg-lr-3e5-bs-512)         | [valentwin-academic_papers-n-100-hn-10-selective-neg-lr-3e5-bs-512-ep-10](https://huggingface.co/albertus-andito/valentwin-academic_papers-n-100-hn-10-selective-neg-lr-3e5-bs-512-ep-10)       |
| books               | [valentwin-books-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-books-n-100-hn-10-selective-neg-lr-3e5-bs-512)                             | [valentwin-books-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-books-n-100-hn-10-selective-neg-lr-3e5-bs-512)                                       |
| cosmetics           | [valentwin-cosmetics-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-cosmetics-n-100-hn-10-selective-neg-lr-3e5-bs-512)                     | [valentwin-cosmetics-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-cosmetics-n-100-hn-10-selective-neg-lr-3e5-bs-512)                               |
| movies              | [valentwin-movies-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-movies-n-100-hn-10-selective-neg-lr-3e5-bs-512)                           | [valentwin-movies-n-100-hn-10-selective-neg-lr-3e5-bs-512-ep-10](https://huggingface.co/albertus-andito/valentwin-movies-n-100-hn-10-selective-neg-lr-3e5-bs-512-ep-10)                         |
| restaurants         | [valentwin-restaurants-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-restaurants-n-100-hn-10-selective-neg-lr-3e5-bs-512)                 | [valentwin-restaurants-n-100-hn-10-selective-neg-lr-3e5-bs-512](https://huggingface.co/albertus-andito/valentwin-restaurants-n-100-hn-10-selective-neg-lr-3e5-bs-512)                           |
</details>

## How to Run

### 1. Sample Data
The sample data is in the `data` folder. If you don't already have it and want to sample the data yourself, you can run the `split_and_sample_datasets.py` from the `scripts` folder, e.g.:
```bash
python scripts/split_and_sample_datasets.py --dataset_dir_paths data/magellan/books/formatted \
    --sample_dataset_dir_paths data/magellan/books/sample \
    --sample_sizes 100 \
    --split_ratio 0.4 0.2 0.4 \
    --include_all_samples \
    --drop_duplicates \
    --seed 42
```
or just run the bash script
```shell
cd scripts/shell
sh sample_data.sh
```

### 2. Generate Contrastive Data
The contrastive data is in the `data` folder. If you don't already have it and want to generate the contrastive data yourself, you can run the `generate_contrastive_data.py` from the `scripts` folder, e.g.:
```bash
python scripts/generate_contrastive_data.py --input_dir_paths data/magellan/books/sample/100-train \
    --output_dir_paths data/magellan/books/contrastive-selective/100 \
    --hard_neg_size 10 \
    --with_col_table_names \
    --use_selective_negatives \
    --pretrained_model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --pooling cls \
    --device cuda:0 \
    --num_values_per_item 1
```
or just run the bash script
```shell
cd scripts/shell
sh generate_contrastive_data.sh
```

### 3. Train the Model
We have provided the trained models in the link in the `Trained Models` section that you can directly use.
However, you can also train the model yourself.
To train the model, you can run the `train_valentwin.py` from the `scripts` folder, e.g.:
```bash
python scripts/train_valentwin.py --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --train_file data/magellan/books/contrastive-selective/100/train.csv \
    --validation_file data/magellan/books/contrastive-selective/100/val.csv \
    --eval_file data/magellan/books/contrastive-selective/100/test.csv \
    --output_dir scripts/result/valentwin-books \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --label_names [] \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --metric_for_best_model accuracy \
    --load_best_model_at_end \
    --fp16 \
    --use_in_batch_instances_as_negatives \
    --restrictive_in_batch_negatives \
```
or you can follow the multi-GPU training example in the `scripts/shell/train_valentwin.sh`.

Once the model is finished training, you need to convert the model from SimCSE to HuggingFace type model by running the `simcse_to_huggingface_and_push.py` script.

### 4. Run Matching
To use the model to match the schemas, you can run the `valentwin-batch-matching.py` from the `scripts` folder, e.g.:
```bash
python scripts/valentwin-batch-matching.py \
    --pretrained_model_names_or_paths scripts/result/valentwin-books \
    --measures euc \
    --column_name_weights 0.4 \
    --column_name_measures euc \
    --holistic \
    --tables_root_dir data/magellan/books/sample/100-test \
    --output_root_dir data/magellan/books/output/100 \
    --device cuda:0
```
or you can follow the example in the `scripts/shell/valentwin-matching.sh`.

### 5. Evaluate Matching
To evaluate the matching results, you can run the `calculate_metrics.py` from the `scripts` folder, e.g.:
```bash
python scripts/calculate_metrics.py \
    --input_dir_path data/magellan/books/output/100 \
    --output_file_path data/magellan/books/metrics/100.csv \
    --ground_truth_file_path data/magellan/books/ground-truth-mapping/ground-truth.csv \
    --do_annotate_tp_fp \
    --split_by_column_types \
    --parallel_workers -1
```

## How to Reproduce Experiments

Each subfolder in the `experiments` folder contains the scripts to reproduce the experiments with the results in the paper.
It is assumed that you have already sampled the data and generated the contrastive data as described in the `How to Run` section.


## Acknowledgements
This repository is largely based on [Valentine](https://github.com/delftdata/valentine).
We also include code from [SimCSE](https://github.com/princeton-nlp/SimCSE) with modifications for the model training.
The code for the competitor methods are also taken from their respective repositories: [ALITE](https://github.com/northeastern-datalab/alite) and [Starmie](https://github.com/megagonlabs/starmie/tree/main).