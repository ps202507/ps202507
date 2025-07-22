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
