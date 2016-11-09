#!/usr/bin/env python3

import os
import csv
import re
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
Score = namedtuple('Score', ['id', 'num_correct', 'pct_correct'])

# only consider completed responses
complete_responses = []
for response in responses:
    if int(response['lastpage']) == 45: # 45 total pages in survey
        complete_responses.append(response)

# count correct responses
for response in complete_responses:

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
            num_correct=num_correct,
            pct_correct=pct_correct
        )
    )

print(scores)
