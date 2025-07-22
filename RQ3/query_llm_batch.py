import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from typing import Union, List
import sys
import concurrent
import multiprocessing as mp
import logging
from openai import OpenAI
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import torch
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from utils import utils
from collections import Counter

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))  
        self.rank = [1] * size  

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x]) 
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)

def get_dataset(file_path, split, batch_size=16):
    data_type = ""
    if file_path.endswith(".csv"):
        data_type = "csv"
    elif file_path.endswith(".jsonl") or file_path.endswith(".json"):
        data_type = "json"
    roles = load_dataset(data_type, data_files={"full": file_path})
    if split == "full":
        return roles['full']
    train_roles = roles['full'].shuffle(seed=42).select(range(batch_size))
    test_list = []
    for test in roles['full']:
        if test in train_roles:
            continue
        test_list.append(test)
    return train_roles, test_list

def query_workers(datas, save_path, api_url, rank):
    client = OpenAI(
        api_key="EMPTY",
        base_url=api_url
    )
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            start_idx = len(f.readlines())
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        start_idx = 0
    fp = open(save_path, "a")
    for idx, data in enumerate(tqdm(datas, desc=f"Rank {rank}")):
        if idx < start_idx:
            continue
        try:
            chat_response = client.chat.completions.create(
                model=args.model_path,
                messages=[
                    {"role": "system", "content": data['instruction']},
                    {"role": "user", "content": data['prompt']}
                ],
                temperature=0.0,
                max_tokens=512
            )
            completion = chat_response.choices[0].message.content
            tmp = {"instruction": data['instruction'], "prompt": data['prompt'], "completion": completion}
        except Exception as e:
            print(e)
            continue
        fp.write(json.dumps(tmp, ensure_ascii=False) + "\n")
        fp.flush()

def ask_vlm(data, n_threads, save_dir, api_urls):
    pool = mp.Pool(n_threads)
    data_chunks = np.array_split(data, n_threads)
    res_list = []
    for i, data_chunk in enumerate(data_chunks):
        save_path = f"{save_dir}/tmp/rank_{i}.jsonl"
        kwds = {
            "datas": data_chunk,
            "save_path": save_path,
            "api_url": api_urls[i % len(api_urls)],
            "rank": i
        }
        if n_threads >= 1:
            res_list.append(pool.apply_async(query_workers, kwds=kwds))
        else:
            query_workers(**kwds)
    pool.close()
    pool.join()
    [r.get() for r in res_list]

def get_top_k_mutation_results(mutation_res_csv, top_k=10):
    csv_df = pd.read_csv(mutation_res_csv)
    df_list = np.array(csv_df).tolist()
    df_list.sort(key=lambda x: x[4], reverse=True)
    result = []
    for data in df_list:
        if len(result) >= top_k:
            break
        if data[1] in result:
            continue
        result.append(data[1])
    return result

def get_diverse_mutation_results(mutation_res_csv, top_k=10):
    csv_df = pd.read_csv(mutation_res_csv)
    df_list = np.array(csv_df).tolist()
    uf = UnionFind(len(df_list))
    for data in df_list:
        if data[2] != "init_seed":
            uf.union(int(data[2]), data[0])
    sampled_root_node = []
    df_list.sort(key=lambda x: x[4], reverse=True)
    result = []
    for data in df_list:
        if len(result) >= top_k:
            break
        if data[1] in result:
            continue
        if data[2] != "init_seed":
            if uf.find(int(data[2])) not in sampled_root_node:
                sampled_root_node.append(uf.find(int(data[2])))
                result.append(data[1])
                continue
        else:
            if uf.find(data[0]) not in sampled_root_node:
                sampled_root_node.append(uf.find(data[0]))
                result.append(data[1])
                continue

    return result

def combine_json(experiment_path):
    json_list = os.listdir(os.path.join(experiment_path, "tmp"))
    combine_res = []
    for idx in range(len(json_list)):
        with open(os.path.join(experiment_path, "tmp", f"rank_{idx}.jsonl"), "r") as f:
            for l in f.readlines():
                cur_content = json.loads(l)
                combine_res.append(cur_content)
    with open(os.path.join(experiment_path, "results.jsonl"), "w") as f:
        for item in combine_res:
            f.write(json.dumps(item) + "\n")
    print("Combine Done!")
    print(f"Results are saved in {os.path.join(experiment_path, 'results.jsonl')}")

def parse_args(args):
    parser = argparse.ArgumentParser(description="Adversarial Suffix Optimization")
    parser.add_argument(
        "--model_path", 
        type=str, 
        help="Victim model path."
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="data/sharegpt-test.jsonl", 
        help="System Prompts Dataset Path."
    )
    parser.add_argument(
        "--n_threads", 
        type=int, 
        default=128, 
        help="Number of Threads"
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="vllm", 
        help="Victim model deployment type (vllm server or local)."
    )
    parser.add_argument(
        "--attack_inst_type", 
        type=str, 
        default="wo_suffix", 
        help="Attack Instruction Type (wo_suffix: without adversarial suffix, w_suffix:with adversarial suffix)."
    )
    parser.add_argument(
        "--adversarial_suffix_file_path", 
        type=str, 
        default=None, 
        help="Adversarial suffix optimization file path. (Only needs in w_suffix mode)"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=10, 
        help="Top-k mutation results. (Only needs in wo_suffix mode)"
    )
    parser.add_argument(
        "--mutation_results_path", 
        type=str, 
        default=None, 
        help="Mutation results path. (Only needs in wo_suffix mode)"
    )
    parser.add_argument(
        "--mutation_sample_type", 
        type=str, 
        default="diverse", 
        help="Mutation results path. (Only needs in wo_suffix mode)"
    )
    parser.add_argument(
        "--save_ans_path", 
        type=str, 
        default=None, 
        help="Save Answer Path."
    )
    parser.add_argument(
        "--debugpy", 
        action="store_true",
        help="Debug Mode"
    )
    args = parser.parse_args(args)
    return args

if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)
    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    if hasattr(args, 'debugpy') and args.debugpy:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        debugpy.breakpoint()
    
    model_name_dict = {"llama-3.1-8B-Instruct": "llama3.1", "Qwen2.5-7B-Instruct": "qwen2.5",
                       "vicuna-7b-v1.5": "vicuna"}
    dataset_name_dict = {"data/unnatural-test.jsonl": "unnatural", "data/awesome.csv": "awesome",
                         "data/sharegpt-test.jsonl": "sharegpt"}
    
    model_path = args.model_path
    model_name = model_path.split("/")[-1]
    dataset_path = args.dataset_path
    dataset_name = dataset_name_dict[dataset_path]

    logging.info(f"Current Processing dataset is {dataset_name}")
    n_threads = args.n_threads
    api_urls = ["http://0.0.0.0:8080/v1","http://0.0.0.0:8081/v1"]
    train_num = 16
    train_split, test_split = get_dataset(dataset_path, 'split', train_num)
    extraction_prompt = [] 
    if args.attack_inst_type == "wo_suffix":
        if len(extraction_prompt) > 0:
            logging.warning("Please Comfirm your experiments settings!!")
            input()
        else:
            logging.info("Begin to load mutation results...")
            logging.info(f"mutation_results_path: {args.mutation_results_path}")
            pre_top_k = args.top_k
            mutation_results = []
            mutation_res_csv = args.mutation_results_path
            if len(extraction_prompt) == 0:
                if args.mutation_sample_type == "diverse":
                    logging.info("Begin to get diverse mutation results...")
                    extraction_prompt = get_diverse_mutation_results(mutation_res_csv, pre_top_k)
                elif args.mutation_sample_type == "top_k":
                    extraction_prompt = get_top_k_mutation_results(mutation_res_csv, pre_top_k)
                else:
                    raise ValueError("Mutation Sample Type Not Supported!")
    elif args.attack_inst_type == "w_suffix":
        logging.info("Begin to load adversarial suffix optimization results...")
        logging.info(f"adversarial_suffix_file_path: {args.adversarial_suffix_file_path}")
        adversarial_suffix_file_path = args.adversarial_suffix_file_path
        with open(adversarial_suffix_file_path, 'r') as f:
            adversarial_suffix_optimization_result = f.readlines()
            for res in adversarial_suffix_optimization_result:
                res = json.loads(res)
                prompt = res['prompt'].strip()
                trigger = res['trigger'].strip()
                optimization_prompt = prompt + " " + trigger
                extraction_prompt.append(optimization_prompt)
    else:
        raise ValueError("Attack Instruction Type Not Supported!")

    datas = []
    for i in test_split:
        for j in extraction_prompt:
            if 'instruction' in i.keys():
                datas.append({"instruction":i['instruction'], "prompt":j})
            elif 'prompt' in i.keys():
                datas.append({"instruction":i['prompt'], "prompt":j})

    logging.info(len(datas))
    t = time.localtime()
    t = t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec
    if args.save_ans_path is None:
        save_ans_path = f"experiment/{model_name}/{dataset_name}/{t[0]}.{t[1]}.{t[2]}/{t[3]}.{t[4]}.{t[5]}/"
    else:
        save_ans_path = f"experiment/{model_name}/{dataset_name}/{t[0]}.{t[1]}.{t[2]}/{args.save_ans_path}/"

    if os.path.exists(save_ans_path):
        logging.info(f"Experiment {save_ans_path} is already exists!")
        logging.info(f"It might be completed. Please check the path!")
        exit(0)

    logging.info(f"Current Processing dataset is {dataset_name}")
    logging.info(f"Current Processing model is {model_name}")

    if args.model_type == "vllm":
        logging.info("Begin to ask VLM...")
        ask_vlm(datas, n_threads, save_ans_path, api_urls)
        logging.info("VLM Query Done!\n\n")
    else:
        raise ValueError("Model Type Not Supported!")

    logging.info("Begin to combine results...")
    combine_json(save_ans_path)