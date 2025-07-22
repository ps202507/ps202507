echo "evaluate_extraction...."
export CUDA_VISIBLE_DEVICES=0
evaluate_results=xxx
python evaluate_extraction.py \
    --extraction_file_path ${evaluate_results} \
