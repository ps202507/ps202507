import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class ModelFactory():
    def __init__(self):
        self.MODEL_CONF = {}
        self._register_model_config('vicuna', 'vicuna-7b-v1.5', 32000)
        self._register_model_config('llama3.1-8b',"llama-3.1-8B-Instruct", 128256)
        self._register_model_config('llama3.2-1b',"llama-3.2-1B-Instruct", 128256)
        self._register_model_config('qwen2.5-7b',"Qwen2.5-7B-Instruct", 151936)
        self._register_model_config('qwen2.5-1.5b',"Qwen2.5-1.5B-Instruct", 151936)

    def _register_model_config(self, name, alias, vocab_size):
        self.MODEL_CONF[name] = {'alias': alias, 'vocab_size': vocab_size}

    def get_vocab_size(self, name):
        return self.MODEL_CONF[name]['vocab_size']

    def get_tokenizer(self, name):
        return AutoTokenizer.from_pretrained(self.MODEL_CONF[name]['alias'])

    def get_model(self, name):
        return AutoModelForCausalLM.from_pretrained(self.MODEL_CONF[name]['alias'], torch_dtype=torch.bfloat16, device_map="auto")