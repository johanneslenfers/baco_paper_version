#!/usr/bin/python
import math
import numpy as np
import os
import sys
import warnings
from collections import OrderedDict
import csv 

sys.path.append(".")
# from baco import optimizer  # noqa
#from baco import run # noqa
from baco import run

rewrites = {
    0: "id",
    1: "splitJoinRule",
    2: "fuseReduceMap",
    3: "mapGlobal(0)",
    4: "mapGlobal(1)",
    5: "mapSeqCompute",
    6: "mapLocal(0)",
    7: "mapLocal(1)",
    8: "mapWorkGroup(0)",
    9: "mapWorkGroup(1)",
}
rewrite_performance = {}

def initialize(csv_file_path: str):

    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)


        # get best performance for each made rewrite 
        current_rewrite = "[]"
        current_rewrite_performance = 2147483647
        for row in csv_reader:

            if row['rewrite'] != current_rewrite:

                # add rewrite to performance dict 
                if current_rewrite_performance == 2147483647:

                    rewrite_performance[current_rewrite] = {
                        'runtime': current_rewrite_performance, 
                        'valid': False
                        }

                else:
                    rewrite_performance[current_rewrite] = {
                        'runtime': current_rewrite_performance, 
                        'valid': True
                    }

                current_rewrite = row['rewrite']
                current_rewrite_performance = 2147483647

            else: 
                if row['error-level'] == "None":
                    if float(row['runtime']) < current_rewrite_performance:
                        current_rewrite_performance = float(row['runtime'])


def parse_rewrite(sample: dict) -> str:
    rewrite = "["
    previous_id = False

    # get type of entry (rewrite vs. index)
    for entry in sample:
        if "R" in entry:
            # filter out id rewrite 
            if sample[entry].item() == 0:
                previous_id = True
            # add rewrite to rewrite sequence 
            else:
                rewrite += f"({rewrites[sample[entry].item()]}, "
                previous_id = False
        else:
            # add index to rewrite sequence 
            if not previous_id:
                rewrite += f"{sample[entry]}),"

    # remove last comma if present and add closing bracket 
    if len(rewrite) > 1:
        rewrite = f"{rewrite[:-1]}]"
    else:
        rewrite += "]"

    return rewrite

# cost function 
def rise(sample: dict):
       
    performance = {}
    rewrite = parse_rewrite(sample=sample)
    try:
        performance = rewrite_performance[rewrite]

    except Exception as e:
        performance = {
            'runtime': 2147483647,
            'valid': False
        }

    return performance

def main():
    # initialize('rewrite_performance_long.csv')
    initialize('rewrites.csv')
    parameters_file = "rewriting.json"
    run.optimize(parameters_file, rise)

if __name__ == "__main__":
    main()
