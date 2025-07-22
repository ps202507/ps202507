import random
import numpy as np
from Template.fuzzer import GPTFuzzer, PromptNode


class SelectPolicy:
    def __init__(self, fuzzer: GPTFuzzer):
        self.fuzzer = fuzzer

    def select(self) -> PromptNode:
        raise NotImplementedError(
            "SelectPolicy must implement select method.")

    def update(self, prompt_nodes: 'list[PromptNode]'):
        pass


class MCTSExploreSelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: GPTFuzzer = None, ratio=0.5, alpha=0.1, beta=0.2):
        super().__init__(fuzzer)
        self.step = 0  
        self.mctc_select_path: 'list[PromptNode]' = []  
        self.last_choice_index = None  
        self.rewards = []  
        self.ratio = ratio 
        self.alpha = alpha  
        self.beta = beta   
    def select(self) -> PromptNode:
        self.step += 1
        if len(self.fuzzer.prompt_nodes) > len(self.rewards):
            self.rewards.extend(
                [0 for _ in range(len(self.fuzzer.prompt_nodes) - len(self.rewards))])

        self.mctc_select_path.clear()
        cur = max(
            self.fuzzer.initial_prompts_nodes,
            key=lambda pn:
            self.rewards[pn.index] / (pn.visited_num + 1) +
            self.ratio * np.sqrt(2 * np.log(self.step) /
                                 (pn.visited_num + 0.01))
        )
        self.mctc_select_path.append(cur)
        while len(cur.child) > 0:
            if np.random.rand() < self.alpha:
                break
            cur = max(
                cur.child,
                key=lambda pn:
                self.rewards[pn.index] / (pn.visited_num + 1) +
                self.ratio * np.sqrt(2 * np.log(self.step) /
                                     (pn.visited_num + 0.01))
            )
            self.mctc_select_path.append(cur)
        for pn in self.mctc_select_path:
            pn.visited_num += 1
        self.last_choice_index = cur.index
        return cur

    def update(self, prompt_nodes: 'list[PromptNode]'):
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes])
        last_choice_node = self.fuzzer.prompt_nodes[self.last_choice_index]
        for prompt_node in reversed(self.mctc_select_path):
            reward = succ_num / (len(self.fuzzer.system_prompts)
                                 * len(prompt_nodes))
            self.rewards[prompt_node.index] += reward * \
                max(self.beta, (1 - 0.1 * last_choice_node.level))

