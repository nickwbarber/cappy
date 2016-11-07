#!/usr/bin/env python3

import os
import csv
import re
import argparse

parser = argparse.ArgumentParser(description='statistically analyzes simultaneous dichotic listening responses from CSV files')
parser.add_argument('source', type=str, help='the source CSV file containing results from dichotic listening test')

args = parser.parse_args()
source = args.source

with open(source) as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)
    for i in headers:
        print('hit')
