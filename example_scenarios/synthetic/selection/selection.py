#!/usr/bin/python3.8
import math

import os
import sys
import warnings
from collections import OrderedDict
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# from baco import run

try:
    from baco.run import optimize  # noqa
except:
    sys.path.append(".")
    from baco.run import optimize  # noqa


def selection(X):

    rewriting: str = X["rewriting"]
    # print("rewriting: " + str(rewriting))

    cost: float = 0.0
    for value in rewriting:
        if value == 'A':
            cost += 0.8
        elif value == 'B':
            cost += 0.9
        elif value == 'C':
            cost += 1
        else:
            cost += 0
    
    cost *= 1/(''.join(rewriting).find('A') + 2)

    valid = True

    # fake constraint 
    # if ''.join(rewriting).count('C') > 1:
    #     valid = False

    return {"runtime": cost, "Valid": valid}


def main():
    parameters_file = (
        "example_scenarios/synthetic/selection/selection.json"
    )
    optimize(parameters_file, selection)
    print("End of known constraints")


if __name__ == "__main__":
    main()

