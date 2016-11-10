#!/usr/bin/env python3

import os
import csv
import argparse
import re
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
total_responses = len(responses) # tally for later

scores = []

# only consider completed responses
complete_responses = []
for response in responses:
    if int(response['lastpage']) == 45: # total pages in survey
        complete_responses.append(response)
responses.clear() # clear, now that we're done
total_complete_responses = len(complete_responses)

# only consider responses with high pretest scores
valid_responses = []
pre_test_length = len(answer_key.pretest)
for response in complete_responses:
    pre_total_correct = 0
    for column in response:
        if column in answer_key.pretest:
            q_code = column
            if answer_key.verify_answer(q_code, response[q_code]):
                pre_total_correct += 1
    if pre_total_correct/pre_test_length >= 5/6:
        valid_responses.append(response)
complete_responses.clear()
total_valid_responses = len(valid_responses)

# print response statistics, then clear from memory
print('{} responses'.format(total_responses))
print('{} complete responses'.format(total_complete_responses))
print('{} valid, complete responses'.format(total_valid_responses))
print('')


# score format declaration
Score = namedtuple(
    'Score',
    ['id',
    'age',
    'sex',
    'languages',
    'left_correct',
    'right_correct',
    'total_correct',
    'pct_left_correct',
    'pct_right_correct',
    'pct_correct'
    ]
    )

# for calculation of ratios
total_prompts = len(answer_key.key)
left_prompts = len([ x for x in answer_key.key if 'l' in x.lower() ])
right_prompts = len([ x for x in answer_key.key if 'r' in x.lower() ])

# TODO: count correct both

for response in valid_responses:

    # count correct responses
    total_correct = 0
    left_correct = 0
    right_correct = 0
    for column in response:
        if column in answer_key.key:
            q_code = column
            if answer_key.verify_answer(q_code, response[q_code]):
                total_correct += 1
                if 'l' in q_code.lower(): left_correct += 1
                if 'r' in q_code.lower(): right_correct += 1
    pct_correct = round(total_correct/total_prompts,2)
    pct_left_correct = round(left_correct/left_prompts,2)
    pct_right_correct = round(right_correct/right_prompts,2)

    # tabulate scores
    scores.append(
        Score(
            id=response['id'],
            age=response['Age'],
            sex=response['Sex'],
            languages=response['Languages'],
            left_correct=left_correct,
            right_correct=right_correct,
            total_correct=total_correct,
            pct_left_correct=pct_left_correct,
            pct_right_correct=pct_right_correct,
            pct_correct=pct_correct
            )
        )

valid_responses.clear() # clear from memory once done

for score in scores:
    print(score)
