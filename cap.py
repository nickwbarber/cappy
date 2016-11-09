#!/usr/bin/env python3

import os
import csv
import argparse
from collections import namedtuple
import answer_key # 'key' as dict, defs verify_answer()

# executable with command-line arguments
parser = argparse.ArgumentParser(
    description=(
    '''statistically analyzes simultaneous dichotic listening responses
     from CSV files'''
    )
)
parser.add_argument(
    'response_set',
    type=str,
    help='''the source CSV file containing results from dichotic listening
     test'''
)
args = parser.parse_args()

# collect responses, one per row
with open(args.response_set) as response_csv:
    responses = [ response for response in csv.DictReader(response_csv) ]

scores = []

# score format declaration
Score = namedtuple('Score', ['id', 'age', 'sex', 'languages', 'num_correct', 'pct_correct'])

# only consider completed responses
complete_responses = []
for response in responses:
    if int(response['lastpage']) == 45: # total pages in survey
        complete_responses.append(response)

# only consider responses with high pretest scores
valid_responses = []
for response in complete_responses:
    pre_num_correct = 0
    for prompt, answer in answer_key.pretest.items():
        if response[prompt].lower() == answer.lower():
            pre_num_correct += 1
    if pre_num_correct/len(answer_key.pretest) >= 5/6:
        valid_responses.append(response)

# print response statistics, then clear from memory
print('{} responses'.format(len(responses)))
print('{} complete responses'.format(len(complete_responses)))
print('{} valid, complete responses'.format(len(valid_responses)))
print('')
responses.clear()
complete_responses.clear()

# count correct responses
for response in valid_responses:
    num_correct = 0
    for column in response:
        if column in answer_key.key:
            q_code = column
            if answer_key.verify_answer(q_code, response[q_code]):
                num_correct += 1
    pct_correct = round(num_correct/len(answer_key.key),2)
    scores.append(
        Score(
            id=response['id'],
            age=response['Age'],
            sex=response['Sex'],
            languages=response['Languages'],
            num_correct=num_correct,
            pct_correct=pct_correct
        )
    )

for score in scores:
    print(score)
