import logging
import time
import csv
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .mutator import Mutator, MutatePolicy
    from .mutator_openai_model import LLMMutator, OpenAIMutatePolicy
    from .selection import SelectPolicy
from ..llm import LocalLLM
import warnings

class PromptNode:
    def __init__(self,
                 fuzzer: 'GPTFuzzer',
                 prompt: str,
                 response: str = None,
                 results: 'list[int]' = None,
                 parent: 'PromptNode' = None,
                 mutator: 'LLMMutator' = None):
        self.fuzzer: 'GPTFuzzer' = fuzzer
        self.prompt: str = prompt
        self.response: str = response
        self.results: 'list[int]' = results
        self.visited_num = 0  
        self.parent: 'PromptNode' = parent
        self.mutator: 'LLMMutator' = mutator
        self.child: 'list[PromptNode]' = [] 
        self.level: int = 0 if parent is None else parent.level + 1  
        self._index: int = None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        if self.parent is not None:
            self.parent.child.append(self)

    @property
    def num_jailbreak(self):
        return sum(self.results)

    @property
    def num_reject(self):
        return len(self.results) - sum(self.results)

    @property
    def num_query(self):
        return len(self.results)


class GPTFuzzer:
    def __init__(self,
                 target: 'LocalLLM',
                 initial_seed: 'list[str]',
                 mutate_policy: 'OpenAIMutatePolicy',
                 select_policy: 'SelectPolicy',
                 system_prompts: 'list[str]',
                 max_query: int = -1,
                 max_jailbreak: int = -1,
                 max_reject: int = -1,
                 max_iteration: int = -1,
                 energy: int = 1,
                 result_file: str = None,
                 generate_in_batch: bool = False,
                 evaluate_metric: str = 'rougel',
                 ):
        self.system_prompts = system_prompts
        self.target: LocalLLM = target
        self.prompt_nodes: 'list[PromptNode]' = [
            PromptNode(self, prompt) for prompt in initial_seed
        ]
        self.initial_prompts_nodes = self.prompt_nodes.copy()

        for i, prompt_node in enumerate(self.prompt_nodes):
            prompt_node.index = i

        self.mutate_policy = mutate_policy
        self.select_policy = select_policy
        self.current_query: int = 0
        self.current_jailbreak: int = 0
        self.current_reject: int = 0
        self.current_iteration: int = 0
        self.max_extraction_count: int = 0
        self.max_query: int = max_query
        self.max_jailbreak: int = max_jailbreak
        self.max_reject: int = max_reject
        self.max_iteration: int = max_iteration
        self.evaluate_metric: str = evaluate_metric
        self.energy: int = energy
        if result_file is None:
            result_file = f'results.csv'
        self.raw_fp = open(result_file, 'a', buffering=1)
        self.writter = csv.writer(self.raw_fp)
        self.writter.writerow(
            ['index', 'prompt', 'parent', 'results', 'successful_rate'])
        self.generate_in_batch = False
        if generate_in_batch is True:
            self.generate_in_batch = True
            if isinstance(self.target, LocalLLM):
                warnings.warn("IMPORTANT! Hugging face inference with batch generation has the problem of consistency due to pad tokens. We do not suggest doing so and you may experience (1) degraded output quality due to long padding tokens, (2) inconsistent responses due to different number of padding tokens during reproduction. You should turn off generate_in_batch or use vllm batch inference.")
        self.setup()

    def setup(self):
        self.mutate_policy.fuzzer = self
        self.select_policy.fuzzer = self
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s %(message)s', datefmt='[%H:%M:%S]')

    def is_stop(self):
        checks = [
            ('max_query', 'current_query'),
            ('max_jailbreak', 'current_jailbreak'),
            ('max_reject', 'current_reject'),
            ('max_iteration', 'current_iteration'),
        ]
        return any(getattr(self, max_attr) != -1 and getattr(self, curr_attr) >= getattr(self, max_attr) for max_attr, curr_attr in checks)

    def run(self):
        for prompt_node in self.prompt_nodes:
            logging.info("Evaluating Oringinal Seed")
            self.evaluate([prompt_node])
            self.writter.writerow([prompt_node.index, prompt_node.prompt,
                        "init_seed", prompt_node.results, sum(prompt_node.results) / len(self.system_prompts)])
            self.current_jailbreak = prompt_node.num_jailbreak
            self.current_reject = prompt_node.num_reject
            self.log()
        logging.info("Fuzzing started!")
        try:
            while not self.is_stop():
                seed = self.select_policy.select()
                mutated_results = self.mutate_policy.mutate_single(seed)
                self.evaluate(mutated_results)
                self.update(mutated_results)
                self.log()
        except KeyboardInterrupt:
            logging.info("Fuzzing interrupted by user!")
        logging.info("Fuzzing finished!")
        self.raw_fp.close()

    def evaluate(self, prompt_nodes: 'list[PromptNode]'):
        logging.info("Current Prompt Node")
        for prompt_node in prompt_nodes:
            logging.info(prompt_node.prompt)
        for prompt_node in prompt_nodes:
            successful_extraction = self.target.generate_answers(self.system_prompts, prompt_node.prompt, metric=self.evaluate_metric)
            prompt_node.results = successful_extraction

    def update(self, prompt_nodes: 'list[PromptNode]'):
        self.current_iteration += 1
        for prompt_node in prompt_nodes:
            self.max_extraction_count = prompt_node.num_jailbreak
            prompt_node.index = len(self.prompt_nodes)
            self.prompt_nodes.append(prompt_node) 
            self.writter.writerow([prompt_node.index, prompt_node.prompt,
                                    prompt_node.parent.index, prompt_node.results, sum(prompt_node.results) / len(self.system_prompts)])
            self.current_jailbreak = prompt_node.num_jailbreak
            self.current_reject = prompt_node.num_reject
            self.current_query += prompt_node.num_query
        self.select_policy.update(prompt_nodes)

    def log(self):
        logging.info(
            f"Iteration {self.current_iteration}: {self.current_jailbreak} jailbreaks, {self.current_reject} rejects, {self.current_query} queries")
