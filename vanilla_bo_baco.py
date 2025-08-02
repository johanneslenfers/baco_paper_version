#!/usr/bin/python
import math
import numpy as np
import os
import sys
import warnings
from collections import OrderedDict


sys.path.append(".")
#from baco import optimizer  # noqa
from baco import run # noqa
#import run
import random

# from typing import Dict

# a synthetic cost function 
# Example usage and test function
def example_compiler_objective(x: dict[str, float]) -> float:
    """
    Example compiler optimization function.
    Simulates optimizing tile sizes with cache constraints and performance cliffs.
    """
    
    # Simulate some compiler optimization landscape
    # This mimics tile size optimization with cache effects
    
    # Convert [0,1] to tile sizes (e.g., 1 to 128)
    A = x['A']
    B = x['B']
    
    # Simulate performance calculation
    performance = 1.0
    
    # Performance generally increases with tile size
    performance *= (1.0 + 0.1 * A / 128.0)
        
    # But drops dramatically if not power of 2 (vectorization penalty)
    # if not (A & (A - 1) == 0):  # Not power of 2
        # performance *= 0.7
        
    # Cache constraint: performance cliff if product of dimensions > 8192
    if np.prod(A * B) > 8192:
        performance *= 0.3
    
    # Add some noise
    performance += np.random.normal(0, 0.05)
   
    return performance


def main():
    parameters_file = "vanilla_bo_baco.json"
    run.optimize(parameters_file, example_compiler_objective)


if __name__ == "__main__":
    main()
