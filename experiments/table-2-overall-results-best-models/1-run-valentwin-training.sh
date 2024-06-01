NUM_GPU=8
PORT_ID=$(expr $RANDOM + 1000)

torchrun --nproc-per-node=$NUM_GPU --rdzv-endpoint=localhost:$PORT_ID ../../scripts/train_valentwin.py \
    --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --train_file ../../data/alite/25ksep11/contrastive-selective/100/train.csv \
    --validation_file ../../data/alite/25ksep11/contrastive-selective/100/val.csv \
    --eval_file ../../data/alite/25ksep11/contrastive-selective/100/test.csv \
    --output_dir ../result/valentwin-25ksep11-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10 \
    --num_train_epochs 10 \
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
    --report_to wandb \
    --run_name valentwin-25ksep11-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10\
    --fp16 \
    "$@"

torchrun --nproc-per-node=$NUM_GPU --rdzv-endpoint=localhost:$PORT_ID ../../scripts/train_valentwin.py \
    --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --train_file ../../data/alite/500spend/contrastive-selective/100/train.csv \
    --validation_file ../../data/alite/500spend/contrastive-selective/100/val.csv \
    --eval_file ../../data/alite/500spend/contrastive-selective/100/test.csv \
    --output_dir ../result/valentwin-500spend-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10 \
    --num_train_epochs 10 \
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
    --report_to wandb \
    --run_name valentwin-500spend-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10 \
    --fp16 \
    "$@"

torchrun --nproc-per-node=$NUM_GPU --rdzv-endpoint=localhost:$PORT_ID ../../scripts/train_valentwin.py \
    --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --train_file ../../data/alite/1009ipopayments/contrastive-selective/100/train.csv \
    --validation_file ../../data/alite/1009ipopayments/contrastive-selective/100/val.csv \
    --eval_file ../../data/alite/1009ipopayments/contrastive-selective/100/test.csv \
    --output_dir ../result/valentwin-1009ipopayments-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10 \
    --num_train_epochs 10 \
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
    --report_to wandb \
    --run_name valentwin-1009ipopayments-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10 \
    --fp16 \
    "$@"

torchrun --nproc-per-node=$NUM_GPU --rdzv-endpoint=localhost:$PORT_ID ../../scripts/train_valentwin.py \
    --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --train_file ../../data/alite/amo-ame/contrastive-selective/100/train.csv \
    --validation_file ../../data/alite/amo-ame/contrastive-selective/100/val.csv \
    --eval_file ../../data/alite/amo-ame/contrastive-selective/100/test.csv \
    --output_dir ../result/valentwin-amo-ame-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10 \
    --num_train_epochs 10 \
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
    --report_to wandb \
    --run_name valentwin-amo-ame-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10 \
    --fp16 \
    "$@"

torchrun --nproc-per-node=$NUM_GPU --rdzv-endpoint=localhost:$PORT_ID ../../scripts/train_valentwin.py \
    --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --train_file ../../data/alite/chicago_parks/contrastive-selective/100/train.csv \
    --validation_file ../../data/alite/chicago_parks/contrastive-selective/100/val.csv \
    --eval_file ../../data/alite/chicago_parks/contrastive-selective/100/test.csv \
    --output_dir ../result/valentwin-chicago_parks-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10 \
    --num_train_epochs 10 \
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
    --report_to wandb \
    --run_name valentwin-chicago_parks-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10 \
    --fp16 \
    "$@"

torchrun --nproc-per-node=$NUM_GPU --rdzv-endpoint=localhost:$PORT_ID ../../scripts/train_valentwin.py \
    --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --train_file ../../data/alite/cihr/contrastive-selective/100/train.csv \
    --validation_file ../../data/alite/cihr/contrastive-selective/100/val.csv \
    --eval_file ../../data/alite/cihr/contrastive-selective/100/test.csv \
    --output_dir ../result/valentwin-cihr-n-100-hn-10-selective-neginter-lr-3e5-bs-512 \
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
    --report_to wandb \
    --run_name valentwin-cihr-n-100-hn-10-selective-neginter-lr-3e5-bs-512 \
    --fp16 \
    --use_in_batch_instances_as_negatives \
    "$@"

torchrun --nproc-per-node=$NUM_GPU --rdzv-endpoint=localhost:$PORT_ID ../../scripts/train_valentwin.py \
    --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --train_file ../../data/alite/DCMS_NHM_NHM/contrastive-selective/100/train.csv \
    --validation_file ../../data/alite/DCMS_NHM_NHM/contrastive-selective/100/val.csv \
    --eval_file ../../data/alite/DCMS_NHM_NHM/contrastive-selective/100/test.csv \
    --output_dir ../result/valentwin-DCMS_NHM_NHM-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10 \
    --num_train_epochs 10 \
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
    --report_to wandb \
    --run_name valentwin-DCMS_NHM_NHM-n-100-hn-10-selective-noneg-lr-3e5-bs-512-ep-10 \
    --fp16 \
    "$@"

torchrun --nproc-per-node=$NUM_GPU --rdzv-endpoint=localhost:$PORT_ID ../../scripts/train_valentwin.py \
    --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --train_file ../../data/alite/organogram-junior/contrastive-selective/100/train.csv \
    --validation_file ../../data/alite/organogram-junior/contrastive-selective/100/val.csv \
    --eval_file ../../data/alite/organogram-junior/contrastive-selective/100/test.csv \
    --output_dir ../result/valentwin-organogram-junior-n-100-hn-10-selective-neg-lr-3e5-bs-512-ep-10 \
    --num_train_epochs 10 \
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
    --report_to wandb \
    --run_name valentwin-organogram-junior-n-100-hn-10-selective-neg-lr-3e5-bs-512-ep-10 \
    --fp16 \
    --use_in_batch_instances_as_negatives \
    --restrictive_in_batch_negatives \
    "$@"

torchrun --nproc-per-node=$NUM_GPU --rdzv-endpoint=localhost:$PORT_ID ../../scripts/train_valentwin.py \
    --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --train_file ../../data/alite/school_report/contrastive-selective/100/train.csv \
    --validation_file ../../data/alite/school_report/contrastive-selective/100/val.csv \
    --eval_file ../../data/alite/school_report/contrastive-selective/100/test.csv \
    --output_dir ../result/valentwin-school_report-n-100-hn-10-selective-neginter-lr-3e5-bs-512-ep-10 \
    --num_train_epochs 10 \
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
    --report_to wandb \
    --run_name valentwin-school_report-n-100-hn-10-selective-neginter-lr-3e5-bs-512-ep-10 \
    --fp16 \
    --use_in_batch_instances_as_negatives \
    "$@"

torchrun --nproc-per-node=$NUM_GPU --rdzv-endpoint=localhost:$PORT_ID ../../scripts/train_valentwin.py \
    --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --train_file ../../data/alite/stockport_contracts/contrastive-selective/100/train.csv \
    --validation_file ../../data/alite/stockport_contracts/contrastive-selective/100/val.csv \
    --eval_file ../../data/alite/stockport_contracts/contrastive-selective/100/test.csv \
    --output_dir ../result/valentwin-stockport_contracts-n-100-hn-10-selective-neg-lr-3e5-bs-512 \
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
    --report_to wandb \
    --run_name valentwin-stockport_contracts-n-100-hn-10-selective-neg-lr-3e5-bs-512 \
    --fp16 \
    --use_in_batch_instances_as_negatives \
    --restrictive_in_batch_negatives \
    "$@"

torchrun --nproc-per-node=$NUM_GPU --rdzv-endpoint=localhost:$PORT_ID ../../scripts/train_valentwin.py \
    --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --train_file ../../data/magellan/academic_papers/contrastive-selective/100/train.csv \
    --validation_file ../../data/magellan/academic_papers/contrastive-selective/100/val.csv \
    --eval_file ../../data/magellan/academic_papers/contrastive-selective/100/test.csv \
    --output_dir ../result/valentwin-academic_papers-n-100-hn-10-selective-neg-lr-3e5-bs-512-ep-10 \
    --num_train_epochs 10 \
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
    --report_to wandb \
    --run_name valentwin-academic_papers-n-100-hn-10-selective-neg-lr-3e5-bs-512-ep-10 \
    --fp16 \
    --use_in_batch_instances_as_negatives \
    --restrictive_in_batch_negatives \
    "$@"

torchrun --nproc-per-node=$NUM_GPU --rdzv-endpoint=localhost:$PORT_ID ../../scripts/train_valentwin.py \
    --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --train_file ../../data/magellan/books/contrastive-selective/100/train.csv \
    --validation_file ../../data/magellan/books/contrastive-selective/100/val.csv \
    --eval_file ../../data/magellan/books/contrastive-selective/100/test.csv \
    --output_dir ../result/valentwin-books-n-100-hn-10-selective-neg-lr-3e5-bs-512 \
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
    --report_to wandb \
    --run_name valentwin-books-n-100-hn-10-selective-neg-lr-3e5-bs-512 \
    --fp16 \
    --use_in_batch_instances_as_negatives \
    --restrictive_in_batch_negatives \
    "$@"

torchrun --nproc-per-node=$NUM_GPU --rdzv-endpoint=localhost:$PORT_ID ../../scripts/train_valentwin.py \
    --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --train_file ../../data/magellan/cosmetics/contrastive-selective/100/train.csv \
    --validation_file ../../data/magellan/cosmetics/contrastive-selective/100/val.csv \
    --eval_file ../../data/magellan/cosmetics/contrastive-selective/100/test.csv \
    --output_dir ../result/valentwin-cosmetics-n-100-hn-10-selective-neg-lr-3e5-bs-512 \
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
    --report_to wandb \
    --run_name valentwin-cosmetics-n-100-hn-10-selective-neg-lr-3e5-bs-512 \
    --fp16 \
    --use_in_batch_instances_as_negatives \
    --restrictive_in_batch_negatives \
    "$@"

torchrun --nproc-per-node=$NUM_GPU --rdzv-endpoint=localhost:$PORT_ID ../../scripts/train_valentwin.py \
    --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --train_file ../../data/magellan/movies/contrastive-selective/100/train.csv \
    --validation_file ../../data/movies/academic_papers/contrastive-selective/100/val.csv \
    --eval_file ../../data/magellan/movies/contrastive-selective/100/test.csv \
    --output_dir ../result/valentwin-movies-n-100-hn-10-selective-neg-lr-3e5-bs-512-ep-10 \
    --num_train_epochs 10 \
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
    --report_to wandb \
    --run_name valentwin-movies-n-100-hn-10-selective-neg-lr-3e5-bs-512-ep-10 \
    --fp16 \
    --use_in_batch_instances_as_negatives \
    --restrictive_in_batch_negatives \
    "$@"

torchrun --nproc-per-node=$NUM_GPU --rdzv-endpoint=localhost:$PORT_ID ../../scripts/train_valentwin.py \
    --model_name_or_path princeton-nlp/sup-simcse-roberta-base \
    --train_file ../../data/magellan/restaurants/contrastive-selective/100/train.csv \
    --validation_file ../../data/magellan/restaurants/contrastive-selective/100/val.csv \
    --eval_file ../../data/magellan/restaurants/contrastive-selective/100/test.csv \
    --output_dir ../result/valentwin-restaurants-n-100-hn-10-selective-neg-lr-3e5-bs-512 \
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
    --report_to wandb \
    --run_name valentwin-restaurants-n-100-hn-10-selective-neg-lr-3e5-bs-512 \
    --fp16 \
    --use_in_batch_instances_as_negatives \
    --restrictive_in_batch_negatives \
    "$@"


