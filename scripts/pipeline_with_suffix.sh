echo "query_llm.sh"

dataset=xxx
model_path=xxx
adversarial_suffix_file_path=xxxx
save_ans_path=xxx

echo "processing awesome"

python query_llm_batch.py \
    --model_path ${model_path} \
    --dataset_path data/awesome.csv \
    --n_threads 128 \
    --model_type vllm \
    --attack_inst_type w_suffix \
    --adversarial_suffix_file_path ${adversarial_suffix_file_path} \
    --save_ans_path ${save_ans_path} \

echo "Dataset awesome done!"

echo "processing sharegpt"

python query_llm_batch.py \
    --model_path ${model_path} \
    --dataset_path data/sharegpt-test.jsonl \
    --n_threads 128 \
    --model_type vllm \
    --attack_inst_type w_suffix \
    --adversarial_suffix_file_path ${adversarial_suffix_file_path} \
    --save_ans_path ${save_ans_path} \

echo "Dataset sharegpt done!"

echo "processing unnatural"

python query_llm_batch.py \
    --model_path ${model_path} \
    --dataset_path data/unnatural-test.jsonl \
    --n_threads 128 \
    --model_type vllm \
    --attack_inst_type w_suffix \
    --adversarial_suffix_file_path ${adversarial_suffix_file_path} \
    --save_ans_path ${save_ans_path} \

echo "Dataset unnatural done!"

echo "query_llm.sh done"