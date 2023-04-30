# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse

from tqdm import tqdm
import dataset

def evaluate_places(filepath, predicted_places):
  """ Computes percent of correctly predicted birth places.

  Arguments:
    filepath: path to a file with our name, birth place data.
    predicted_places: a list of strings representing the
        predicted birth place of each person.

  Returns:
    (total, correct), floats
  """
  with open(filepath) as fin:
    lines = [x.strip().split('\t') for x in fin]
    if len(lines[0]) == 1:
      print('No gold birth places provided; returning (0,0)')
      return (0,0)
    true_places = [x[1] for x in lines]
    total = len(true_places)
    assert total == len(predicted_places)
    correct = len(list(filter(lambda x: x[0] == x[1],
      zip(true_places, predicted_places))))
    return (float(total),float(correct))

argp = argparse.ArgumentParser()
argp.add_argument('function',
    help="Whether to pretrain, finetune or evaluate a model",
    choices=["pretrain", "finetune", "evaluate"])
argp.add_argument('variant',
    help="Which variant of the model to run ('vanilla')",
    choices=["vanilla"])
argp.add_argument('pretrain_corpus_path',
    help="Path of the corpus to pretrain on", default=None)
argp.add_argument('--reading_params_path',
    help="If specified, path of the model to load before finetuning/evaluation",
    default=None)
argp.add_argument('--writing_params_path',
    help="Path to save the model after pretraining/finetuning", default=None)
argp.add_argument('--finetune_corpus_path',
    help="Path of the corpus to finetune on", default=None)
argp.add_argument('--eval_corpus_path',
    help="Path of the corpus to evaluate on", default=None)
argp.add_argument('--outputs_path', default=None)
args = argp.parse_args()

block_size = 128
text = open(args.pretrain_corpus_path).read()
pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)

assert args.outputs_path is not None
assert args.eval_corpus_path is not None
correct = 0
total = 0
with open(args.outputs_path, 'w') as fout:
    predictions = []
    for line in tqdm(open(args.eval_corpus_path)):
        predictions.append('London')
        fout.write('London' + '\n')
    total, correct = evaluate_places(args.eval_corpus_path, predictions)
if total > 0:
    print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
else:
    print('Predictions written to {}; no targets provided'
            .format(args.outputs_path))
