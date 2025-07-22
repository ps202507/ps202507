echo "query_llm.sh"

dataset=xxx
model_path=xxx
mutation_results_path=xxx
save_ans_path=xxx

echo ${mutation_results_path}
echo ${save_ans_path}
echo ${dataset}

echo "processing awesome"
python query_llm_batch.py \
    --model_path ${model_path} \
    --dataset_path data/awesome.csv \
    --n_threads 128 \
    --model_type vllm \
    --attack_inst_type wo_suffix \
    --top_k 10 \
    --mutation_sample_type diverse \
    --mutation_results_path ${mutation_results_path} \
    --save_ans_path ${save_ans_path}

echo "Dataset awesome done!"

echo "processing sharegpt"

python query_llm_batch.py \
    --model_path ${model_path} \
    --dataset_path data/sharegpt-test.jsonl \
    --n_threads 128 \
    --model_type vllm \
    --attack_inst_type wo_suffix \
    --top_k 10 \
    --mutation_sample_type diverse \
    --mutation_results_path ${mutation_results_path} \
    --save_ans_path ${save_ans_path} \

echo "Dataset sharegpt done!"

echo "processing unnatural"

python query_llm_batch.py \
    --model_path ${model_path} \
    --dataset_path data/unnatural-test.jsonl \
    --n_threads 128 \
    --model_type vllm \
    --attack_inst_type wo_suffix \
    --top_k 10 \
    --mutation_sample_type diverse \
    --mutation_results_path ${mutation_results_path} \
    --save_ans_path ${save_ans_path} \
    
echo "Dataset unnatural done!"

echo "query_llm.sh done"