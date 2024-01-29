#!/usr/bin/python
import run
from tests.aux.functions import *
from typing import List
import numpy as np
import os, sys
import matplotlib.pyplot as matplt
import subprocess
import tests.aux.mdh_conv_n16.config_helper as config_helper
import mdh_cost_function.mdh_cost_function as cost_function

def prepare_experiment(methods: List[str], name: str) -> None:

    config_helper.createConfigFilesMdh(INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2], INPUT_SIZE[3], INPUT_SIZE[4], INPUT_SIZE[5], INPUT_SIZE[6], NUM_OPTIMIZATION_ITERATIONS_PER_RUN, SLURM_ARRAY_TASK_ID, TIME_BUDGET)
    name = LOCATION + name
    subprocess.run(['mkdir', '-p', name])

    for method in methods:
        subprocess.run(['mkdir', '-p', f'{name}/{method}/csv'])

    for method in methods:
        subprocess.run(['mkdir', '-p', f'{name}/{method}/log'])

    return None

def get_cost_function():
    return cost_function.mdh_cost_function(
    "/home/h/hohlmeye/evaluation/mdh_conv_n16/baco_" + SLURM_ARRAY_TASK_ID + "/mdh_cost_function/mdh/cuda/build", 
    "conv", 
    INPUT_SIZE)

def run_method(method: str, name: str, cost_function) -> None:

    # run method 
    run.optimize(f"tests/aux/{name}/{name}_{method}.json", cost_function)

    return None


def run_experiment(methods: List[str], name: str) -> None:
    prepare_experiment(methods, name)
    cf = get_cost_function()
    for method in methods:
        run_method(method, name, cf)
    config_helper.removeConfigFilesMdh()

    return None


# add main definition here 
LOCATION: str = "/scratch/tmp/hohlmeye/"
INPUT_SIZE = [16, 112, 112, 64, 3, 7, 7] 
NUM_OPTIMIZATION_ITERATIONS_PER_RUN = 5000000
# specifies the time budget per run in minutes
TIME_BUDGET = 12 * 60
try:
    SLURM_ARRAY_TASK_ID = sys.argv[1]
except IndexError as e:
    sys.exit("no SLURM_ARRAY_TASK_ID passed")
methods: List[str] = [
    # 'random_sampling', # only for control
    'embedding_random_sampling', # unbiased
    'embedding_random_sampling_biased', # biased
    'opentuner', # unbiased 
    'opentuner_biased', # biased 
    ]

run_experiment(methods, "mdh_conv_n16")