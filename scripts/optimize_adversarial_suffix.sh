echo "Optimization for adversarial suffix attack"

model_path=xxx
export CUDA_VISIBLE_DEVICES=0


echo "Optimization for awesome suffix attack"

python adversarial_suffix_optimization.py \
    --model_path ${model_path} \
    --dataset_path data/awesome.csv \

echo "awesome suffix Done!"


echo "Optimization for unnatural suffix attack"
python adversarial_suffix_optimization.py \
    --model_path ${model_path} \
    --dataset_path data/unnatural-test.jsonl

echo "unnatural suffix Done!"


echo "Optimization for sharegpt suffix attack"
python adversarial_suffix_optimization.py \
    --model_path ${model_path} \
    --dataset_path data/sharegpt-test.jsonl \

echo "sharegpt suffix Done!"
