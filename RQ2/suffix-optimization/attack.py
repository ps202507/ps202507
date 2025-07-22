import random
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from ModelFactory import ModelFactory
from utils import utils

class HotFlip:
    def __init__(self, trigger_token_length=6, shadow_model='gpt2', step=50, tokenizer=None, model=None, template=None,
                 init_triggers='', label_slice=50,
                 init_step=None):
       
        self.device = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.target_model = shadow_model       
        modelFactory = ModelFactory()
        self.model = modelFactory.get_model(shadow_model) if model is None else model
        self.tokenizer = modelFactory.get_tokenizer(shadow_model) if tokenizer is None else tokenizer
        self.vocab_size = modelFactory.get_vocab_size(shadow_model)
        self.embedding_weight = self.get_embedding_weight()
        self.step = step
        self.init_step = init_step if init_step is not None else self.step
        self.user_prefix = ''
        self.trigger_tokens = self.tokenizer.encode(("x " * trigger_token_length).strip(), add_special_tokens=False)
        self.label_slice = label_slice
        self._nonascii_toks = self.get_nonascii_toks(self.tokenizer, 'cpu')
        self.best_trigger_tokens = self.trigger_tokens
        self.best_score_value = 0

    def get_nonascii_toks(self, tokenizer, device='cpu'):
        def is_ascii(s):
            return s.isascii() and s.isprintable()
        ascii_toks = []
        for i in range(3, tokenizer.vocab_size):
            if not is_ascii(tokenizer.decode([i])):
                ascii_toks.append(i)
        if tokenizer.bos_token_id is not None:
            ascii_toks.append(tokenizer.bos_token_id)
        if tokenizer.eos_token_id is not None:
            ascii_toks.append(tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None:
            ascii_toks.append(tokenizer.pad_token_id)
        if tokenizer.unk_token_id is not None:
            ascii_toks.append(tokenizer.unk_token_id)
        return torch.tensor(ascii_toks, device=device)

    def get_embedding_weight(self):
        for module in self.model.modules():
            if not isinstance(module, torch.nn.Embedding): continue
            if module.weight.shape[0] != self.vocab_size: continue
            module.weight.requires_grad = True
            return module.weight.detach()

    def get_triggers_grad(self):
        for module in self.model.modules():
            if not isinstance(module, torch.nn.Embedding):
                continue
            if module.weight.shape[0] != self.vocab_size:
                continue
            return module.weight.grad[self.trigger_tokens]

    def decode_triggers(self):
        return self.user_prefix + self.tokenizer.decode(self.best_trigger_tokens)
    

    def make_target_chat(self, target_text, first_word_extraction_basic_prompt, triggers):
        if type(triggers) != str:
            triggers = self.tokenizer.decode(triggers)          
        target = [
            {"role": "system", "content": target_text},
            {"role": "user", "content": first_word_extraction_basic_prompt + " " + triggers},
            {"role": "assistant", "content": target_text},
        ]
        target = self.tokenizer.apply_chat_template(target)
        non_label = [
            {"role": "system", "content": target_text},
            {"role": "user", "content": first_word_extraction_basic_prompt + " " + triggers},
        ]
        non_label = self.tokenizer.apply_chat_template(non_label, add_generation_prompt=True)
        encoded_label = [-100] * len(non_label) + target[len(non_label):len(non_label) + self.label_slice]
        target = target[:len(encoded_label)]  
        if len(target) != len(encoded_label):
            if len(target) > len(encoded_label):
                target = target[:len(encoded_label)]
            else:
                encoded_label = encoded_label[:len(target)]
        label = torch.tensor([encoded_label], device=self.device)
        lm_input = torch.tensor([target], device=self.device)
        return lm_input, label

    def batch_compute_loss(self, target_texts, inst_extraction_prompt, triggers, require_grad=False): 
        if type(target_texts) != str:
            if 'instruction' in target_texts[0].keys():
                target_texts = [i['instruction'] for i in target_texts]
            elif 'prompt' in target_texts[0].keys():
                target_texts = [i['prompt'] for i in target_texts]
            else:
                raise ValueError("No instruction or prompt key in target_texts")
            
        if type(triggers) != str:
            triggers = self.tokenizer.decode(triggers)
        targets = []
        no_labels = []
        for idx, inst in enumerate(target_texts):
            target = [
                {"role": "system", "content": inst},
                {"role": "user", "content": inst_extraction_prompt + " " + triggers},
                {"role": "assistant", "content": inst},
            ]
            target = self.tokenizer.apply_chat_template(target, tokenize=False)
            no_label = [
                {"role": "system", "content": inst},
                {"role": "user", "content": inst_extraction_prompt + " " + triggers},
            ]
            no_label = self.tokenizer.apply_chat_template(no_label, tokenize=False, add_generation_prompt=True)
            targets.append(target)
            no_labels.append(no_label)
        encoded_inputs = self.tokenizer(targets, add_special_tokens=False)
        encoded_no_labels = self.tokenizer(no_labels, add_special_tokens=False)
        lm_inputs = []
        lm_labels = []
        for lm_input, no_label in zip(encoded_inputs['input_ids'], encoded_no_labels['input_ids']):
            label = [-100] * len(no_label) + lm_input[len(no_label): len(no_label) + self.label_slice]
            lm_input = lm_input[:len(label)]
            lm_inputs.append(lm_input)
            lm_labels.append(label)
        max_len = max(map(len, lm_inputs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id       
        padding_lm_inputs = [[self.tokenizer.pad_token_id] * (max_len - len(x)) + x for x in lm_inputs]
        padding_lm_labels = [[-100] * (max_len - len(x)) + x for x in lm_labels]
        tensor_labels = torch.tensor(padding_lm_labels).to(self.model.device)
        tensor_inputs = torch.tensor(padding_lm_inputs).to(self.model.device)
        mask = (tensor_inputs != self.tokenizer.pad_token_id).float().to(self.model.device)
        outputs = self.model(input_ids=tensor_inputs, attention_mask=mask, labels=tensor_labels)
        loss = outputs.loss
        if require_grad:
            loss.backward()
        return loss

    def compute_loss(self, target_texts, trigger_tokens, extraction_prompt,
                     require_grad=False):
        total_loss = 0
        for index, text in enumerate(target_texts):
            try:
                if "instruction" in text.keys():
                    lm_input, label = self.make_target_chat(text['instruction'], extraction_prompt, trigger_tokens)
                elif 'prompt' in text.keys():
                    lm_input, label = self.make_target_chat(text['prompt'], extraction_prompt, trigger_tokens)
                else:
                    raise ValueError("No instruction or prompt key in target_texts")
                lm_input = lm_input.to(self.model.device)
                label = label.to(self.model.device)
                print(f"Input shape: {lm_input.shape}, Label shape: {label.shape}")
                if lm_input.shape[1] != label.shape[1]:
                    min_len = min(lm_input.shape[1], label.shape[1])
                    lm_input = lm_input[:, :min_len]
                    label = label[:, :min_len]
                loss = self.model(lm_input, labels=label)[0] / len(target_texts)
                if require_grad:
                    loss.backward()
                total_loss += loss.item()
            except Exception as e:
                print(f"Error processing sample {index}: {str(e)}")
                continue
        return total_loss

    def hotflip_attack(self, averaged_grad, increase_loss=False, num_candidates=10):
        averaged_grad = averaged_grad
        embedding_matrix = self.embedding_weight
        averaged_grad = averaged_grad.unsqueeze(0)
        gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",
                                                     (averaged_grad, embedding_matrix))
        gradient_dot_embedding_matrix *= -1
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().squeeze().cpu().numpy()

    def generate_answers(self, target_texts, extraction_basic_prompt, triggers):
        from utils import utils
        evaluator = utils.Evaluator()
        if 'instruction' in target_texts[0].keys():
            prompts = [
                [
                    {"role": "system", "content": target_text['instruction']},
                    {"role": "user", "content": extraction_basic_prompt + " " + self.tokenizer.decode(triggers)}
                ]
                for target_text in target_texts
            ]
        elif 'prompt' in target_texts[0].keys():
            prompts = [
                [
                    {"role": "system", "content": target_text['prompt']},
                    {"role": "user", "content": extraction_basic_prompt + " " + self.tokenizer.decode(triggers)}
                ]
                for target_text in target_texts
            ]
        else:
            raise ValueError("No instruction or prompt key in target_texts")     
        mask_token_id = self.tokenizer.eos_token_id
        inputs = [self.tokenizer.apply_chat_template(x, add_generation_prompt=True) for x in prompts]
        maxlen = max(map(len, inputs))
        inputs = [[mask_token_id] * (maxlen - len(x)) + x for x in inputs]
        iids = torch.tensor(inputs).to(self.model.device)
        mask = (iids != mask_token_id).float().to(self.model.device)
        out = self.model.generate(
            iids,
            max_new_tokens=512,
            attention_mask=mask,
            do_sample=False
        )
        completion_raws = [
            self.tokenizer.decode(x[maxlen:], skip_special_tokens=True) for x in out
        ]
        results = []
        for idx, c in enumerate(completion_raws):
            results.append({'instruction': prompts[idx][0]['content'], 'completion': c})
            print("-"*50)
            print("\n\ninstruction:")
            print(prompts[idx][0]['content'])
            print("\n\ncompletion:")
            print(c)
        res = evaluator.evaluate(results, level='em')
        res = torch.mean(res).item()
        return res

    def get_max_tokens_count(self, target_texts, optimization_prompt):
        max_count = 0
        for i in target_texts['instruction']:
            no_label = [
                {"role": "system", "content": i},
                {"role": "user", "content": optimization_prompt}
            ]
            label = [
                {"role": "system", "content": i},
                {"role": "user", "content": optimization_prompt},
                {"role": "assistant", "content": i}
            ]
            no_label = self.tokenizer.apply_chat_template(no_label, add_generation_prompt=True)
            label = self.tokenizer.apply_chat_template(label)
            token_count = len(label) - len(no_label)
            if token_count > max_count:
                max_count = token_count

        return max_count

    def replace_triggers(self, target_texts, optimization_prompt, allow_nonascii_tokens=True): 
        print(f"init_triggers:{self.decode_triggers()}")
        print(f"slice: {self.label_slice}")
        idx_loss = 1
        while idx_loss <= 1:
            print(f"Enter next iteration :{idx_loss}")     
            token_flipped = True
            best_loss = 114514
            iteration = 0
            while token_flipped:
                token_flipped = False     
                with torch.set_grad_enabled(True):
                    self.model.zero_grad()
                    self.compute_loss(target_texts, self.trigger_tokens, optimization_prompt, require_grad=True)
                candidates = self.hotflip_attack(self.get_triggers_grad(), num_candidates=30)
                lowest_loss_trigger_tokens = deepcopy(self.trigger_tokens)
                pos_list = [j for j in range(len(self.trigger_tokens))]
                for i in range(len(self.trigger_tokens)):
                    cur_cand_pos = random.sample(pos_list, 1)[0]
                    pos_list.remove(cur_cand_pos)
                    for cand in tqdm(candidates[cur_cand_pos],
                                     desc=f'Calculating candidates loss of {cur_cand_pos} ...'):
                        if not allow_nonascii_tokens:
                            if cand in self._nonascii_toks:
                                continue
                        candidate_trigger_tokens = deepcopy(self.trigger_tokens)
                        candidate_trigger_tokens[cur_cand_pos] = cand
                        self.model.zero_grad()
                        with torch.no_grad():
                            loss = self.batch_compute_loss(target_texts, optimization_prompt, candidate_trigger_tokens)
                        if best_loss <= loss:
                            continue
                        token_flipped = True
                        best_loss = loss
                        lowest_loss_trigger_tokens = deepcopy(candidate_trigger_tokens)
                        break

                    if token_flipped:
                        break
                self.trigger_tokens = deepcopy(lowest_loss_trigger_tokens)
                iteration += 1
                if iteration % 10 == 0:
                    em_score = self.generate_answers(target_texts, optimization_prompt, self.trigger_tokens)
                    if em_score > self.best_score_value:
                        self.best_trigger_tokens = self.trigger_tokens
                        self.best_score_value = em_score
                if token_flipped:
                    print(f"Loss: {best_loss}, triggers:{repr(self.tokenizer.decode(self.trigger_tokens))}")
                    print(f"Best Selected score: {self.best_score_value}, triggers:{repr(self.decode_triggers())}")
                else:
                    em_score = self.generate_answers(target_texts, optimization_prompt, self.trigger_tokens)
                    if em_score > self.best_score_value:
                        self.best_trigger_tokens = self.trigger_tokens
                        self.best_score_value = em_score
                    print(f"\nNo improvement, ending iteration")
            idx_loss += 1
            print(f"Best Substring score: {self.best_score_value}, triggers:{repr(self.decode_triggers())}")

