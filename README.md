This repository provides the code and datasets to reproduce the experiments in our paper. The framework consists of two main stages:

1. **Prompt Template Optimization** (RQ1)  
2. **Adversarial Suffix Optimization** (RQ2)  

We further provide:
- **RQ3**: End-to-end Evaluation  
- **RQ4**: Hyperparameter Study  
- **RQ5**: Transferability Experiments

---

## 📦 Dataset Download

Please download the following datasets and place them in the `data/` folder:

- `ShareGPT`: [https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main]
- `Awesome`: [https://huggingface.co/datasets/fka/awesome-chatgpt-prompts]
- `Unnatural`: [https://huggingface.co/datasets/mrm8488/unnatural-instructions-full]

Each dataset should be a `.json` or `.csv` file with the instruction/prompt structure described in our paper.

---

## ⚙️ Environment Setup

- Python 3.10+
- Install dependencies via conda:

```bash
conda env create -f environment.yml
conda activate promptstitch
```

---

## 🤖 RQ1: Prompt Template Optimization

We use **Prompt Fuzzing** to mutate the initial prompt seeds.

### Usage

1. Modify `prompt_fuzzer.py`:
   - Set `model_path` to your HuggingFace-style LLM path.
   - Set `api_key` if using OpenAI APIs.

2. Run the script:

```bash
python prompt_fuzzer.py
```

---

## 🤖 RQ2: Adversarial Suffix Optimization

We use gradient-guided optimization to search for effective suffixes to be appended to the prompts.

### Usage

1. Modify `scripts/optimize_adversarial_suffix.sh`:
   - Set `model_path` to your LLM checkpoint path.

2. Run the script:

```bash
bash scripts/optimize_adversarial_suffix.sh
```

This will generate suffixes for each prompt template.

---

## 🤖 RQ3: Inference and Final Extraction

We infer answers from the target model to assess attack effectiveness.

### Step 1: Model Deployment (vLLM)

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve model_path --tensor-parallel-size 1 --port 8081
```

### Step 2: Run Inference

Modify the following scripts:
- `scripts/pipeline_with_suffix.sh` (for suffix-optimized prompts)
- `scripts/pipeline_without_suffix.sh` (for template-only prompts)

Update the variables:

```bash
dataset=xxx
model_path=xxx
adversarial_suffix_file_path=xxx  # only in pipeline_with_suffix
mutation_results_path=xxx         # only in pipeline_without_suffix
save_ans_path=xxx
```

Run:

```bash
bash scripts/pipeline_with_suffix.sh
bash scripts/pipeline_without_suffix.sh
```

### Step 3: Select Final Answers

We select the most probable completions (1 from 10 without suffix, 1 from 5 with suffix), similar to Carlini et al.

Run:

```bash
bash scripts/evaluate_extraction.sh
```

Modify the variable:

```bash
evaluate_results="path1.jsonl,path2.jsonl"
```

### Step 4: Generate Final Outputs

Modify `scripts/get_final_answers.sh`:

```bash
em_optimization_results_path=xxx       # from pipeline_with_suffix
rougel_optimization_results_path=xxx   # from pipeline_without_suffix
```

Run:

```bash
bash scripts/get_final_answers.sh
```

---

## 🔧 RQ4: Hyperparameter Study

To analyze the effect of suffix length:

1. Modify the suffix length parameter in `optimize_adversarial_suffix.sh`.
2. Repeat the process described in RQ2.
3. Compare EM results across different lengths.

---

## 🔁 RQ5: Transferability Experiments

We evaluate transferability across:
- **Datasets (Same Model)**
- **Models (Same Dataset)**
- **Joint (Different Models + Datasets)**

### How to Run

1. Generate prompts and suffixes on one source setting.
2. Change inference targets (test dataset and/or model path).
3. Use the same pipeline in RQ3 to produce results.

Examples:
- **Dataset transfer**: `train=ShareGPT` → `test=Awesome`
- **Model transfer**: `train=Qwen` → `test=LLaMA`
- **Joint transfer**: `train=(Qwen, ShareGPT)` → `test=(LLaMA, Unnatural)`
