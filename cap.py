#!/usr/bin/env python3

import os
import csv
import argparse
import re
from collections import namedtuple
import numpy
from scipy import stats
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

# only consider completed responses
complete_responses = []
for response in responses:
    if int(response['lastpage']) == 47: # total pages in survey
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

# print response statistics
print('{} responses'.format(total_responses))
print('{} complete responses'.format(total_complete_responses))
print('{} valid, complete responses'.format(total_valid_responses))
print('')


scores = []

# score format declaration
Score = namedtuple(
    'Score',
    ['id',
        'age',
        'sex',
        'languages',
        'left_correct',
        'right_correct',
        'both_correct',
        'total_correct',
        'pct_left_correct',
        'pct_right_correct',
        'pct_both_correct',
        'pct_correct'
        ]
    )


left_prompts = [x for x in answer_key.key if 'l' in x.lower()]
right_prompts = [x for x in answer_key.key if 'r' in x.lower()]


# for calculation of ratios
total_prompts = len(answer_key.key)
num_paired_prompts = len(answer_key.prompt_groups)
num_left_prompts = len(left_prompts)
num_right_prompts = len(right_prompts)

# TODO: count correct both
AnswerPair = namedtuple('AnswerPair', ['left', 'right'])

for response in valid_responses:

    # count correct responses
    total_correct = 0
    left_correct = 0
    right_correct = 0
    both_correct = 0

    answers = {}
    for column in response:
        if column in answer_key.key:
            q_code = column
            answers.update({q_code: response[q_code]})

    for q_code, answer in answers.items():
        if answer.lower() == answer_key.key.get(q_code).lower():
            total_correct += 1
            if 'l' in q_code.lower(): left_correct += 1
            if 'r' in q_code.lower(): right_correct += 1

    for prompt_group in answer_key.prompt_groups:
        group_left_correct = False
        group_right_correct = False
        for q_code in answers:
            if prompt_group in q_code:
                if answers[q_code].lower() == answer_key.key.get(q_code).lower():
                    if 'l' in q_code.lower():
                        group_left_correct = True
                    if 'r' in q_code.lower():
                        group_right_correct = True
        if group_left_correct and group_right_correct: both_correct += 1

    pct_correct = round(total_correct/total_prompts,2)
    pct_left_correct = round(left_correct/num_left_prompts,2)
    pct_right_correct = round(right_correct/num_right_prompts,2)
    pct_both_correct = round(both_correct/num_paired_prompts,2)

    # tabulate scores
    scores.append(
        Score(
            id=response['id'],
            age=response['Age'],
            sex=response['Sex'],
            languages=response['Languages'],
            left_correct=left_correct,
            right_correct=right_correct,
            both_correct=both_correct,
            total_correct=total_correct,
            pct_left_correct=pct_left_correct,
            pct_right_correct=pct_right_correct,
            pct_both_correct=pct_both_correct,
            pct_correct=pct_correct
            )
        )

valid_responses.clear() # clear from memory once done

for score in scores:
    print(score)

#TODO: make this a function to be used for each test condition
#def ttest():
monos = []
multis = []
for score in scores:
    if int(score.languages) > 1: multis.append(score)
    else: monos.append(score)
#print(stats.normaltest([ x.pct_correct for x in monos ]))
#print(stats.normaltest([ x.pct_correct for x in multis ]))
mono_mean = numpy.mean([ x.pct_correct for x in monos ])
multi_mean = numpy.mean([ x.pct_correct for x in multis ])
mono_std = numpy.std([ x.pct_correct for x in monos ])
multi_std = numpy.std([ x.pct_correct for x in multis ])
mono_var = mono_std ** 2
multi_var = multi_std ** 2
tvalue = (
    abs(
        (mono_mean - multi_mean)
        / (
            (mono_var / len(monos))
            + (multi_var / len(multis))
            ) ** (1/2) # sqrt
        )
    )

print('mean of monos = {}'.format(mono_mean))
print('mean of multis = {}'.format(multi_mean))
print('std of monos = {}'.format(mono_std))
print('std of multis = {}'.format(multi_std))
print('var of monos = {}'.format(mono_var))
print('var of multis = {}'.format(multi_var))
print('t-value = {}'.format(tvalue))

lang_df = len(monos) + len(multis) - 2
print('degrees of freedom = {}'.format(lang_df))
