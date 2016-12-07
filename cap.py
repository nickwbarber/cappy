#!/usr/bin/env python3

import os
import csv
import argparse
import re
from collections import namedtuple
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import answer_key


# executable with command-line arguments
parser = argparse.ArgumentParser(
    description=(
        '''statistically analyzes responses from simultaneous '''
        '''dichotic listening experiments using t-tests.'''
        )
    )
parser.add_argument(
    '-d', '--dry-run',
    dest='dry_run',
    default=False,
    action='store_true',
    help='''displays statistics to terminal.'''
    )
parser.add_argument(
    '--indep-vars',
    dest='indep_vars',
    type=str,
    required=True,
    nargs='+',
    help='''the independent variable to consider. '''
        '''Can be either 'languages' or 'sex'.'''
    )
parser.add_argument(
    '--conditions',
    type=str,
    required=True,
    nargs='+',
    help='''the experimental condition to consider. '''
        '''Can be either 'left', 'right', 'both', or 'any'.'''
        ''''left' considers all left correct, 'both' considers each prompt '''
        '''in which both left and right were correct, '''
        '''and 'any' considers all correct answers in total.'''
    )
parser.add_argument(
    '--response-set',
    dest='response_set',
    required=True,
    type=str,
    help='''the source CSV file containing results '''
        '''from dichotic listening test'''
    )
parser.add_argument(
    '--ttest',
    dest='ttest_file',
    type=str,
    default=False,
    help='''the destination to output a stats file, if specified'''
    )
parser.add_argument(
    '--raw-scores',
    dest='raw_scores',
    type=str,
    default=False,
    help='''outputs individual scores to CSV file.'''
    )

args = parser.parse_args()

valid_indep_vars = [
    'languages',
    'sex'
    ]
valid_conditions = [
    'left',
    'right',
    'both',
    'any'
    ]

indep_vars = args.indep_vars
conditions = args.conditions
response_set = args.response_set
ttest_file = args.ttest_file
raw_scores = args.raw_scores
dry_run = args.dry_run

# validate indep. and dep. variables
indep_vars_validity = True
for i in indep_vars:
    if i not in valid_indep_vars:
        print("Error: '{}' not a valid independent variable to consider.".format(i))
        indep_vars_validity = False
conditions_validity = True
for i in conditions:
    if i not in valid_conditions:
        print("Error: '{}' not a valid condition to consider.".format(i))
        condition_validity = False
if not valid_indep_vars or not valid_conditions:
    quit()

Stat = namedtuple(
    'Stat', [
        'response_set',
        'indep_var',
        'condition',
        'sample1_name',
        'sample2_name',
        'sample1_mean',
        'sample2_mean',
        'sample1_std',
        'sample2_std',
        'sample1_var',
        'sample2_var',
        'df',
        'tvalue'
        ]
    )

def ttest(scores=False, indep_var=False, condition=False):
    sample1 = []
    sample2 = []
    if indep_var == 'languages':
        sample1_name = 'monolinguals'
        sample2_name = 'multilinguals'
        for score in scores:
            if int(score.languages) > 1: sample2.append(score)
            else: sample1.append(score)
    elif indep_var == 'sex':
        sample1_name = 'males'
        sample2_name = 'females'
        for score in scores:
            if score.sex.lower() == 'male': sample1.append(score)
            elif score.sex.lower() == 'female': sample2.append(score)

    #print(stats.normaltest([ x.pct_left_correct for x in sample1 ]))
    #print(stats.normaltest([ x.pct_left_correct for x in sample2 ]))
    if condition == 'left':
        sample1_mean = np.mean([ x.pct_left_correct for x in sample1 ])
        sample2_mean = np.mean([ x.pct_left_correct for x in sample2 ])
        sample1_std = np.std([ x.pct_left_correct for x in sample1 ])
        sample2_std = np.std([ x.pct_left_correct for x in sample2 ])
    elif condition == 'right':
        sample1_mean = np.mean([ x.pct_right_correct for x in sample1 ])
        sample2_mean = np.mean([ x.pct_right_correct for x in sample2 ])
        sample1_std = np.std([ x.pct_right_correct for x in sample1 ])
        sample2_std = np.std([ x.pct_right_correct for x in sample2 ])
    elif condition == 'both':
        sample1_mean = np.mean([ x.pct_both_correct for x in sample1 ])
        sample2_mean = np.mean([ x.pct_both_correct for x in sample2 ])
        sample1_std = np.std([ x.pct_both_correct for x in sample1 ])
        sample2_std = np.std([ x.pct_both_correct for x in sample2 ])
    elif condition == 'any':
        sample1_mean = np.mean([ x.pct_correct for x in sample1 ])
        sample2_mean = np.mean([ x.pct_correct for x in sample2 ])
        sample1_std = np.std([ x.pct_correct for x in sample1 ])
        sample2_std = np.std([ x.pct_correct for x in sample2 ])

    sample1_var = sample1_std ** 2
    sample2_var = sample2_std ** 2
    df = len(sample1) + len(sample2) - 2
    tvalue = (
        abs(
            (sample1_mean - sample2_mean)
            / ((sample1_var / len(sample1))
                + (sample2_var / len(sample2))
                ) ** (1/2) # sqrt
            )
        )

    return Stat(
        response_set=response_set,
        indep_var=indep_var,
        condition=condition,
        sample1_name=sample1_name,
        sample2_name=sample2_name,
        sample1_mean=sample1_mean,
        sample2_mean=sample2_mean,
        sample1_std=sample1_std,
        sample2_std=sample2_std,
        sample1_var=sample1_var,
        sample2_var=sample2_var,
        df=df,
        tvalue=tvalue
        )

# collect responses, one per row
with open(response_set, newline='') as response_csv:
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

# for calculation of ratios
left_prompts = [x for x in answer_key.key if 'l' in x.lower()]
right_prompts = [x for x in answer_key.key if 'r' in x.lower()]

total_prompts = len(answer_key.key)
num_paired_prompts = len(answer_key.prompt_groups)
num_left_prompts = len(left_prompts)
num_right_prompts = len(right_prompts)

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
        'both_empty',
        'pct_left_correct',
        'pct_right_correct',
        'pct_both_correct',
        'pct_correct',
        'pct_both_empty'
        ]
    )

for response in valid_responses:

    key = answer_key.key

    # count correct responses
    total_correct = 0
    left_correct = 0
    right_correct = 0
    both_correct = 0
    both_empty = 0

    answers = {}
    for column in response:
        if column in key:
            q_code = column
            answers.update({q_code: response[q_code]})

    for q_code, answer in answers.items():
        if answer.lower() == key.get(q_code).lower():
            total_correct += 1
            if 'l' in q_code.lower(): left_correct += 1
            if 'r' in q_code.lower(): right_correct += 1

    for prompt_group in answer_key.prompt_groups:
        group_left_correct = False
        group_right_correct = False
        group_left_empty = False
        group_right_empty = False
        for q_code in answers:
            if prompt_group in q_code:
                if 'l' in q_code.lower():
                    if answers[q_code].lower() == key.get(q_code).lower():
                        group_left_correct = True
                    if answers[q_code].lower() == '':
                        group_left_empty = True
                if 'r' in q_code.lower():
                    if answers[q_code].lower() == key.get(q_code).lower():
                        group_right_correct = True
                    if answers[q_code].lower() == '':
                        group_right_empty = True
        if group_left_correct and group_right_correct:
            both_correct += 1
        if group_left_empty and group_right_empty:
            both_empty += 1

    pct_correct = total_correct/total_prompts
    pct_left_correct = left_correct/num_left_prompts
    pct_right_correct = right_correct/num_right_prompts
    pct_both_correct = both_correct/num_paired_prompts
    pct_both_empty = both_empty/num_paired_prompts

    # tabulate scores
    scores.append(
        Score(
            id=response['id'],
            age=response['Age'],
            sex=response['Sex'],
            languages=int(response['Languages']),
            left_correct=int(left_correct),
            right_correct=int(right_correct),
            both_correct=int(both_correct),
            total_correct=int(total_correct),
            both_empty=int(both_empty),
            pct_left_correct=float(pct_left_correct),
            pct_right_correct=float(pct_right_correct),
            pct_both_correct=float(pct_both_correct),
            pct_correct=float(pct_correct),
            pct_both_empty=float(pct_both_empty)
            )
        )
valid_responses.clear() # clear from memory once done
scores = tuple(scores)

num_empty_responses = len(
    [score for score in scores if score.pct_both_empty > (1/3)]
    )
print(
    [score for score in scores if score.pct_both_empty > (1/3)]
    )


# print response statistics
print('{} responses'.format(total_responses))
print('{} complete responses'.format(total_complete_responses))
print('{} valid, complete responses'.format(total_valid_responses))
print('{} valid, complete responses with more than 1/3 empty'.format(num_empty_responses))

if dry_run:
    if ttest_file:
        for indep_var in indep_vars:
            for condition in conditions:
                statistic = ttest(
                    scores=scores,
                    indep_var=indep_var,
                    condition=condition
                    )
                print('')
                print("results file = '{}'".format(statistic.response_set))
                print('independent variable = {}'.format(statistic.indep_var))
                print('condition = {}'.format(statistic.condition))
                print('mean of {} = {}'.format(
                        statistic.sample1_name, statistic.sample1_mean
                        )
                    )
                print('mean of {} = {}'.format(
                        statistic.sample2_name, statistic.sample2_mean
                        )
                    )
                print('std of {} = {}'.format(
                        statistic.sample1_name, statistic.sample1_std
                        )
                    )
                print('std of {} = {}'.format(
                        statistic.sample2_name, statistic.sample2_std
                        )
                    )
                print('var of {} = {}'.format(
                        statistic.sample1_name, statistic.sample1_var
                        )
                    )
                print('var of {} = {}'.format(
                        statistic.sample2_name, statistic.sample2_var
                        )
                    )
                print('t-value = {}'.format(statistic.tvalue))

                print('degrees of freedom = {}'.format(statistic.df))
    if raw_scores:
        print(','.join(Score._fields))
if dry_run: quit()

if ttest_file:
    with open(ttest_file, 'w', encoding='utf-8', newline='') as f:
        csv.writer(f).writerow(Stat._fields)
        for indep_var in indep_vars:
            for condition in conditions:
                statistic = ttest(
                    scores=scores,
                    indep_var=indep_var,
                    condition=condition
                    )
                csv.writer(f).writerow(statistic)
if raw_scores:
    with open(raw_scores, 'w', encoding='utf-8', newline='') as f:
        csv.writer(f).writerow(Score._fields)
        for score in scores:
            csv.writer(f).writerow(score)

else:
    #x = np.array([float(score.pct_correct) for score in scores if int(score.languages) <= 1])
    #y = np.array([float(score.pct_correct) for score in scores if int(score.languages) > 1])

    doof = pandas.DataFrame.from_dict([score._asdict() for score in scores])
    plt.figure()
    #sns.set(style='whitegrid', palette='pastel', color_codes=True)
    sns.set_context(context='poster')
    sns.violinplot(x=doof.languages, y=doof.pct_both_correct, hue=doof.sex, split=True)
    #sns.boxplot(x=doof.languages, y=doof.pct_both_correct)
    plt.show()
    '''
    plt.boxplot([x,y])
    plt.axis([0,3,0,1])
    '''
