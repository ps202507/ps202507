import argparse
import json
from utils import utils
import sys  

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--em_optimization_results_path', type=str, required=True)
    parser.add_argument('--rougel_optimization_results_path', type=str, required=True)
    return parser.parse_args(args)


if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)
    em_optimization_results_path = args.em_optimization_results_path
    rougel_optimization_results_path = args.rougel_optimization_results_path

    with open(em_optimization_results_path, "r") as f:
        em_results = json.load(f)

    with open(rougel_optimization_results_path, "r") as f:
        rougel_results = json.load(f)

    em_results['guesses'].sort(key=lambda x: x['instruction'])
    rougel_results['guesses'].sort(key=lambda x: x['instruction'])

    evaluator = utils.Evaluator()
    final_results = []
    for em, rougel in zip(em_results['guesses'], rougel_results['guesses']):
        assert em['instruction'] == rougel['instruction']

        em_results = em['guess']
        rougel_results = rougel['guess']

        eval_pair = [{'em_res': em_results, 'rouge_res': rougel_results}]
        r = evaluator.evaluate(eval_pair, level='substring')
        
        if r == 1:
            final_results.append({'instruction': em['instruction'], 'guess': em_results})
        
        final_results.append({'instruction': rougel['instruction'], 'guess': rougel_results})

    evaluator.evaluate(final_results, level='substring')
    evaluator.evaluate(final_results, level='em')
    evaluator.evaluate(final_results, level='rouge-l')
    evaluator.evaluate(final_results, level='semantic')
    evaluator.evaluate(final_results, level='close-matching')
    evaluator.evaluate(final_results, level='bleu')
