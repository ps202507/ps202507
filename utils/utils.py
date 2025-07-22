import csv
import re
from torchmetrics import ExtendedEditDistance, CatMetric
from torchmetrics.text import BLEUScore
from torchmetrics.functional.text import bleu_score
import torch
from rouge_score import rouge_scorer

class Evaluator:
    def __init__(self):
        print("Evaluation!!")

    def save_to_csv(self, path, results, triggers):
        with open(path, 'w', newline='') as file:
            fieldnames = ['context', triggers]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)

    def sentence_to_char(self, sentence):
        ret_chars = re.sub('[^a-zA-Z]', '', sentence.lower())
        return ret_chars

    def filter_tokens(self, sentence):
        ret_sentence = re.sub('[^a-zA-Z]', ' ', sentence.lower())
        filtered_sentence = ''.join(self.sentence_to_char(ret_sentence))
        return filtered_sentence

    def evaluate(self, results, level='em', print_answer=True):
        metric = CatMetric()
        keys = list(results[0].keys())
        if level == 'em':
            for result in results:
                target_text = result[keys[0]]
                target = self.filter_tokens(target_text)
                pred = self.filter_tokens(result[keys[1]])
                if target == pred:
                    metric.update(1)
                else:
                    metric.update(0)
                mean = torch.mean(metric.compute())
            if print_answer:
                print(f"em Acc: {mean.item()}")
            return metric.compute()
            
        elif level == 'substring':
            for i, result in enumerate(results):
                target_text = result[keys[0]]
                target = self.filter_tokens(target_text)
                pred = self.filter_tokens(result[keys[1]])
                if target in pred:
                    metric.update(1)
                else:
                    metric.update(0)
                mean = torch.mean(metric.compute())
            if print_answer:
                print(f"s Acc: {mean.item()}")
            return metric.compute()

        elif level == 'edit':
            EDD = ExtendedEditDistance()
            for result in results:
                target_text = result[keys[0]]
                dist = EDD([result[keys[1]]], [target_text])
                metric.update(dist)
            std, mean = torch.std_mean(metric.compute())
            print(f"edit distance mean: {mean.item()}, std: {std.item()}")
            return metric.compute()

        elif level == 'semantic':
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            for result in results:
                target_text = result[keys[0]]
                embedding_1 = model.encode(result[keys[1]], convert_to_tensor=True)
                embedding_2 = model.encode(target_text, convert_to_tensor=True)
                sim = util.pytorch_cos_sim(embedding_1, embedding_2)
                metric.update(sim.to('cpu'))
            std, mean = torch.std_mean(metric.compute())
            if print_answer:
                print(f"semantic mean: {mean.item()}, std: {std.item()}")
            return metric.compute()

        elif level == "close-matching":
            from nltk import sent_tokenize
            for result in results:
                target_text = result[keys[0]]
                pred = result[keys[1]]
                target_sentences = sent_tokenize(target_text)
                pred_matches = all(sent in pred for sent in target_sentences)
                if pred_matches:
                    metric.update(1)
                else:
                    metric.update(0)
            mean = torch.mean(metric.compute())
            if print_answer:
                print(f"Close-matching matches: {mean.item()}")
            return metric.compute()

        elif level == 'bleu':
            for result in results:
                target_text = result[keys[0]]
                dist = bleu_score([result[keys[1]]], [target_text])
                metric.update(1) if dist >= 0.6 else metric.update(0)
            std, mean = torch.std_mean(metric.compute())
            print(f"BLEU mean: {mean.item()}, std: {std.item()}")
            return metric.compute()

        elif level == 'rouge-l':
            for result in results:
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                scores = scorer.score(result[keys[0]].lower(), result[keys[1]].lower())
                if scores['rougeL'].recall >= 0.9:
                    metric.update(1)
                else:
                    metric.update(0)
            std, mean = torch.std_mean(metric.compute())
            if print_answer:
                print(f"ROUGE-L Recall mean: {mean.item()}, std: {std.item()}")
            return metric.compute()
        else:
            raise "Unsupport Metric"
