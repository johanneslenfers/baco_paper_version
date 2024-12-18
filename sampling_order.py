#!/usr/bin/python
import sys
import os
import math
import random
import itertools
import json
import numpy as np
import os
import time
import sys
import warnings
from collections import OrderedDict
import subprocess

sys.path.append(".")

from typing import Dict, List, Tuple, Union, Set, Any, Callable
from dataclasses import dataclass

import baco.plot
import baco.plot.plot_optimization_results
import run
from baco import plot

from interopt.parameter import Param, ParamType, Categorical, Integer, IntExponential
from interopt.parameter import Constraint 
from interopt.definition import ProblemDefinition
from catbench.benchmarks import asum, scal, mm, stencil, kmeans, harris, mttkrp, spmm
from interopt import Study


# Only needed since this is in the same repo as schedgehammer.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import catbench as cb
##############################################################

# type to model dependent groups 
DependencyGroup = Tuple[List[Param], List[Constraint]]

def get_json_base_config(optimization_method: str, 
                         optimization_iterations: int,
                         output_data_file: str,
                         log_file: str,
                         ) -> Dict[str, Any]:
    return {
        "application_name": "sampling_order",
        "optimization_objectives": [
            "runtime"
        ],
        "output_data_file": f"{output_data_file}",
        "log_file": f"{log_file}",
        "epsilon_greedy_threshold": 0,
        "models": {
            "model": "random_forest"
        },
        "design_of_experiment": {
            "doe_type": "random sampling",
            "number_of_samples": 0
        },
        "optimization_method": f"{optimization_method}",
        "optimization_iterations": optimization_iterations,
        "time_budget": 10,
        "input_parameters": {},
    }

def get_dependency_groups(problem_definition: ProblemDefinition) -> List[DependencyGroup]:
    
    params: Union[List[Param], List[Dict]] = problem_definition.search_space.params
    constraints: List[Constraint] = problem_definition.search_space.constraints

    # traverse constraints and group them based on dependent parameters
    dependency_groups: list[DependencyGroup] = []
    constraints_groups: list[tuple[set[str], list[Constraint]]] = []
    for constraint in constraints:

        candidate_groups_index: set[int] = set()
        candidate_groups_counter: int = 0

        for constraints_group in constraints_groups:
            
            for param in constraint.dependent_params:

                if param in constraints_group[0]: 
                    candidate_groups_index.add(candidate_groups_counter)

            candidate_groups_counter += 1

        if len(candidate_groups_index) == 0:
            # if there are no candidates -> create new group
            constraints_groups.append(
                (
                    set(constraint.dependent_params),  
                    [constraint]
                )
            )

        # if there is one candidate -> add
        elif (len(candidate_groups_index) == 1):
            # print('         add to existing groupd')

            constraints_groups[list(candidate_groups_index)[0]][0].update(set(constraint.dependent_params))
            constraints_groups[list(candidate_groups_index)[0]][1].append(constraint)

        # if there are multiple candidates -> merge groups 
        else:
            base_index: int = list(candidate_groups_index)[0]

            # add elements to base group 
            constraints_groups[base_index][0].update(set(constraint.dependent_params))
            constraints_groups[base_index][1].append(constraint)

            candidate_groups_index.remove(base_index)

            # add other group to base group, then delete entry for group 
            for index in candidate_groups_index:
                constraints_groups[base_index][0].update(constraints_groups[index][0])
                constraints_groups[base_index][1].extend(constraints_groups[index][1])
                del constraints_groups[index]

    # get full parameter definitons based on collected names
    # TODO: test this 
    for group in constraints_groups:

        parameter_definitions: List[Param] = []
        for parameter_name in group[0]:
            parameter_definitions.append([param for param in params if param.name == parameter_name][0])

        dependency_groups.append(
            (
                parameter_definitions,
                group[1]
            )
        )

    # for each untouched parameter add one group
    for param in params:

        group_candidate: bool = False
        for group in dependency_groups:

            # check if we have a group for this parameter
            for parameter in group[0]:
                if parameter.name == param.name:
                    group_candidate = True
                    break
            
            if group_candidate:
                break

        if not group_candidate:
            dependency_groups.append(
                (
                    [param],
                    []
                )
            )

    return dependency_groups

# set experiment information
OUTPUT_DIR: str = "example_scenarios/synthetic/sampling_order"

META: Dict[str, Any] = {
    "iterations" : 10000,
    "repetitions" : 1,
}

@dataclass
class Benchmark:
    problem_definition: ProblemDefinition
    cost_function: Callable[[dict[str, Any]], float] 

# TODO: refactor this 
# COST functions 

# ASUM 
# asum_study: Study = cb.benchmark("asum") # type: ignore
def asum_cost_function(configuration: Dict[str, Any]) -> float:

    fids_rise: dict[str, Any]= {
        "iterations": META['iterations'], 
        "timeouts": 60000
    }

    result: float = float(asum_study.query(configuration, fids_rise)["compute_time"]) # type: ignore

    return result


# # does not work with model
# scal_study = Study(benchmark_name='scal', 
#              definition=scal.get_scal_definition(),
#              enable_tabular=True, 
#              enable_model=True,
#              dataset='rtxtitan', 
#              enable_download=True,
#              enabled_objectives=['compute_time'], 
#              port=50051,
#              server_addresses=['localhost']
#              )

# disable model but allow to make a lookup or report an error 

# scal_study: Study = cb.benchmark("scal") # type: ignore
# def scal_cost_function(configuration: Dict[str, Any]) -> float:

#     # print(f"configuration: {configuration}")

#     fids_rise: dict[str, Any] = {
#         "iterations": 10, 
#         "timeouts": 60000
#     }

#     # try catch, allow invalid samples 
#     result: float = float(scal_study.query(configuration, fids_rise)["compute_time"]) # type: ignore 

#     return result

# test = Study(benchmark_name='mm', 
#              definition=mm.get_mm_definition(),
#              enable_tabular=True, 
#              enable_model=True,
#              dataset=None, 
#              enable_download=True,
#              enabled_objectives=['compute_time'], 
#              port=50051,
#              server_addresses=['localhost']
#              )

# # mm_study = cb.benchmark("mm")
# def mm_cost_function(configuration: Dict[str, any]) -> float:

#     fids_rise = {
#         "iterations": META["iterations"], 
#         "timeouts": 600000
#     }

#     result: float = test.query(configuration, fids_rise)["compute_time"]

#     return result

kmeans_study: Study = cb.benchmark("kmeans") # type: ignore
def kmeans_cost_function(configuration: Dict[str, Any]) -> float:

    fids_rise: dict[str, Any] = {
        "iterations": META['iterations'], 
        "timeouts": 60000
    }

    result: float = float(kmeans_study.query(configuration, fids_rise)["compute_time"]) # type: ignore 

    return result

stencil_study: Study = cb.benchmark("stencil") # type: ignore
def stencil_cost_function(configuration: Dict[str, Any]) -> float:

    fids_rise: dict[str, Any] = {
        "iterations": META['iterations'], 
        "timeouts": 60000
    }

    result: float = float(stencil_study.query(configuration, fids_rise)["compute_time"]) # type: ignore

    return result

harris_study: Study = cb.benchmark("harris") # type: ignore
def harris_cost_function(configuration: Dict[str, Any]) -> float:

    fids_rise: dict[str, Any] ={
        "iterations": META['iterations'], 
        "timeouts": 60000
    }

    result: float = float(harris_study.query(configuration, fids_rise)["compute_time"]) # type: ignore

    return result


# TACO cost functions
mttkrp_study: Study = cb.benchmark("mttkrp") # type: ignore
def mttkrp_cost_function(configuration: Dict[str, Any]) -> float:

    fids_taco: dict[str, int] = {
        "iterations": 15,
        "repeats": 5,
        "wait_after_run": 1,
        "wait_between_repeats": 1
    }

    # print(f"configuration: \n")
    # for elem in configuration:
    #     print(f"    {elem}: {str(configuration[elem])}")

    result: float = float(mttkrp_study.query(configuration, fids_taco)["compute_time"]) # type: ignore

    # print(f"result: {result}")

    return result

spmm_study: Study = cb.benchmark("spmm") # type: ignore
def spmm_cost_function(configuration: Dict[str, Any]) -> float:

    fids_taco: dict[str, int] = {
        "iterations": 15,
        "repeats": 5,
        "wait_after_run": 1,
        "wait_between_repeats": 1
    }

    # print(f"configuration: \n")
    # for elem in configuration:
    #     print(f"    {elem}: {str(configuration[elem])}")

    result: float = float(spmm_study.query(configuration, fids_taco)["compute_time"]) # type: ignore
    # print(f"result: {result}")

    return result


def generate_valid_orders(parameters: List[List[str]]) -> List[List[List[str]]]:

    if not parameters:
            return []
    
    sorted_parameters: List[List[str]] = [sorted(inner_list) for inner_list in parameters]
        
    # Generate permutations for each sublist
    sublist_permutations: List[List[Tuple[str, ...]]] = [list(itertools.permutations(sublist)) for sublist in sorted_parameters]
    # sublist_permutations2 = [list(itertools.permutations(sublist)) for sublist in parameters]
        
    # Combine permutations while preserving nesting
    results: List[List[List[str]]] = []
    def combine_permutations(index: int = 0, current: list[list[str]]=[])-> None:
        if index == len(sublist_permutations):
            results.append(current)
            return
        for perm in sublist_permutations[index]:
            combine_permutations(index + 1, current + [list(perm)])
        
    combine_permutations()

    return results

def distribute_constraints(constraints: list[str], order: List[List[str]]) -> Dict[str, List[str]]:

    # collect parameters for constraints 
    constraints_parameters: Dict[str, List[str]] = {}
    for constraint in constraints:
        constraints_parameters[constraint] = []
        for tree in order:
            for param in tree:
                if param in constraint:
                    constraints_parameters[constraint].append(param)

    # now get last involved parameter and map constraint to parameter 
    constraints_distribution: Dict[str, List[str]] = {}

    # initialize constraints 
    for constraint in constraints_parameters:
        constraints_distribution[constraints_parameters[constraint][-1]] = []

    # distribute constraints 
    for constraint in constraints_parameters:
        constraints_distribution[constraints_parameters[constraint][-1]].append(constraint)

    return constraints_distribution

def generate_json(constraints: dict[str, list[str]], 
                  order: list[list[str]], 
                  parameters: list[Param], 
                  method: str,
                  number: int, 
                  benchmark_name: str,
                  output_folder: str,
                  iteration: int) -> dict[str, Any]:

    config: Dict[str, Any] = get_json_base_config(
        optimization_method=method,
        optimization_iterations=META['iterations'],
        output_data_file=f"{output_folder}/order_{number}/{method}/csv/order_{number}_iteration_{iteration}.csv",
        log_file=f"{output_folder}/order_{number}/{method}/log/order_{number}_iteration_{iteration}.log", 
    )

    # set name and output files 
    config['application_name'] = f"sampling_order_{number}"
    # config['output_data_file'] = f"example_scenarios/synthetic/sampling_order/order_{number}/csv/order_{number}_iteration_{iteration}.csv"
    # config['log_file'] = f"example_scenarios/synthetic/sampling_order/order_{number}/log/order_{number}.log"

    # set parameters
    for group in order:
        for param in group:
            for parameter in parameters:
                if parameter.name == param:

                    # TODO: check all parameter types
                    values: list[int] = []
                    parameter_type: str = "ordinal"
                    if parameter.param_type_enum == ParamType.INTEGER:
                        values = [i for i in range(parameter.bounds[0], parameter.bounds[1]+1)] # type: ignore
                    elif parameter.param_type_enum == ParamType.INTEGER_EXP:
                        # get valid values 
                        values: list[int] = [2**i for i in range(int(np.log2(parameter.bounds[0])), int(np.log2(parameter.bounds[1]))+1)] # type: ignore
                    elif parameter.param_type_enum == ParamType.CATEGORICAL:
                        parameter_type = "categorical"
                        values = parameter.categories # type: ignore
                    elif parameter.param_type_enum == ParamType.PERMUTATION:
                        parameter_type = "permutation"
                        # values = [i for i in range(parameter.bounds[0], parameter.bounds[1]+1)] # type: ignore

                        # check and translate constraints 
                        # default is a list 

                        # permutaions in baco 
                        # parameter.length 
                        values = [parameter.length] # type: ignore
                        # "parametrization": "spearman",
                        # "p": {
                        #     "constraints": [
                        #         "((p_1 != 1) & (p_i2 < p_i4) & (p_i0 < p_i2) & (p_i1 < p_i2)) | ((p_i4 < p_i2) & (p_i0 < p_i4) & (p_i1 < p_i4))"
                        #     ],
                        #     "parameter_default": [
                        #         0,
                        #         1,
                        #         2,
                        #         3,
                        #         4
                        #     ],
                        #     "parameter_type": "permutation",
                        #     "parametrization": "spearman",
                        #     "values": [
                        #         5
                        #     ]
                        # },
                        
                    else: 
                        pass

                    # TODO: add permutation parameter

                    # get constaints
                    param_constraints: List[str] = []
                    if param in constraints:
                        # replace all python things with the numexpr equivalent 
                        # TODO: check if these are all 
                        param_constraints = [constraint.replace(" and ", " & ") for constraint in constraints[param]]
                        param_constraints = [constraint.replace(" or ", " | ") for constraint in param_constraints]
                        param_constraints = [constraint.replace(" not ", " ~ ") for constraint in param_constraints]
                    
                    if parameter.param_type_enum == ParamType.PERMUTATION:
                        config["input_parameters"][param] = {
                            "parameter_type": parameter_type,
                            "values": values, 
                            "parameter_default": parameter.default if parameter.default is not None else "",
                            "parametrization": "spearman",
                            "constraints": param_constraints, 
                            "dependencies": [] 
                        }
                    else:
                        config["input_parameters"][param] = {
                            "parameter_type": parameter_type,
                            "values": values, 
                            "parameter_default": parameter.default if parameter.default is not None else "",
                            "constraints": param_constraints, 
                            "dependencies": [] 
                        }
                    
                    # remove default if not present 
                    if parameter.default == None:
                        config["input_parameters"][param].pop("parameter_default")


    # set sampling order 
    config['sampling_order'] = order

    return config

def save_json(configuration: Dict[str, Any], file_name: str) -> None:

    with open(file_name, 'w') as file:
        json.dump(configuration, file, indent=4)
    
    return None

# SETUP Benchmarks and Methods
BENCHMARKS: Dict[str, Benchmark] = {
    # TACO 
    
    # "spmm",
    # "spmm": Benchmark(spmm.get_spmm_definition(), spmm_cost_function),
    # "spmv",
    # "sddmm",
    # "mttkrp": Benchmark(mttkrp.get_mttkrp_definition(), mttkrp_cost_function),
    # "ttv",

    # RISE 
    # "asum" : Benchmark(asum.get_asum_definition(), asum_cost_function),
    # "harris" : Benchmark(harris.get_harris_definition(), harris_cost_function),
    # "kmeans" : Benchmark(kmeans.get_kmeans_definition(), kmeans_cost_function),
    # "stencil" : Benchmark(stencil.get_stencil_definition(), stencil_cost_function),

    # not working  -> model training fails, lookup only is not enough 
    # "scal" : Benchmark(scal.get_scal_definition(), scal_cost_function),
    # "mm":  Benchmark(mm.get_mm_definition(), mm_cost_function),
}

METHODS: List[str] = [
    "embedding_random_sampling",
    "embedding_random_sampling_biased",
    "opentuner",
    "opentuner_biased",
]


def main() -> None:

    for benchmark_name in BENCHMARKS:

        print(f"benchmark_name: {benchmark_name}")

        # create unique output folder for this experiment
        folder_path: str = f"{OUTPUT_DIR}/{benchmark_name}"
        if os.path.isdir(folder_path):
            folder_path = f"{folder_path}_{int(time.time() * 1000)}"

        subprocess.run(["mkdir", "-p", f"{folder_path}"])

        # mk output directory for optimization method 
        # asum:
        #     order_0:
        #         rs
        #         rs_embedding
        #         opentuner
        #         baco 
        #
        #     order_1:
        #         rs
        #         rs_embedding
        #         opentuner
        #         baco 

        # get problem definition and corresponding dependency groups
        problem_definition: ProblemDefinition = BENCHMARKS[benchmark_name].problem_definition
        dependency_groups: List[DependencyGroup] = get_dependency_groups(problem_definition=problem_definition)

        # get orders based on groups 
        parameters: List[List[str]] = [[parameter.name for parameter in group[0]] for group in dependency_groups]
        all_orders: List[List[List[str]]] = generate_valid_orders(parameters=parameters)
        constraints: List[str] = [str(constraint.constraint) for group in dependency_groups for constraint in group[1]]

        print(f"orders: {len(all_orders)}")
        number: int = 0
        for orders in all_orders:

            constraints_distribution: dict[str, list[str]] = distribute_constraints(constraints=constraints, order=orders)

            for method in METHODS:

                # create output directories 
                subprocess.run(["mkdir", "-p", f"{folder_path}/order_{number}/{method}"])
                subprocess.run(["mkdir", "-p", f"{folder_path}/order_{number}/{method}/json"])
                subprocess.run(["mkdir", "-p", f"{folder_path}/order_{number}/{method}/csv"])
                subprocess.run(["mkdir", "-p", f"{folder_path}/order_{number}/{method}/log"])

                for iteration in range(META['repetitions']):

                    # generate and save json as input for the optimizer
                    configuration: Dict[str, Any] = generate_json(constraints=constraints_distribution, 
                                order=orders, 
                                parameters=problem_definition.search_space.params,
                                method=method,
                                number=number,
                                benchmark_name=benchmark_name,
                                output_folder=folder_path,
                                iteration=iteration,
                                )

                    filename: str = f"{folder_path}/order_{number}/{method}/json/order_{number}_iteration_{iteration}.json"
                    save_json(configuration=configuration, file_name=filename) 

                    run.optimize(filename, BENCHMARKS[benchmark_name].cost_function) # type: ignore 


            # plotting for order 
            data_dirs: List[str] = []
            labels: List[str] = [] 
            for method in METHODS:
                data_dirs.append(f"{folder_path}/order_{number}/{method}/csv")
                labels.append(f"{method}")

            plot.plot_optimization_results.plot_regret( # type: ignore
                settings_file=f"{folder_path}/order_{number}/{METHODS[0]}/json/order_{number}_iteration_0.json",
                data_dirs=data_dirs,
                labels=labels,
                minimum=0,
                outfile=f"{folder_path}/order_{number}/order_{number}.pdf",
                title=f"{benchmark_name} order_{number}",
                plot_log=True,
                unlog_y_axis=False,
                budget=None,
                out_dir=f"{folder_path}/order_{number}",
                ncol=2,
                x_label=None,
                y_label=None,
                show_doe=False,
                expert_configuration=None,
            )
            number += 1  

        # now plot for all methods 
        for method in METHODS:
 
            data_dirs: List[str] = [] 
            labels: List[str] = [] 
            
            for number in range(len(all_orders)):
                data_dirs.append(f"{folder_path}/order_{number}/{method}/csv")
                labels.append(f"order_{number}")

            plot.plot_optimization_results.plot_regret( # type: ignore 
                settings_file=f"{folder_path}/order_0/{method}/json/order_0_iteration_0.json",
                data_dirs=data_dirs,
                labels=labels,
                minimum=0,
                outfile=f"{folder_path}/{method}.pdf",
                title=f"{benchmark_name} {method}",
                plot_log=True,
                unlog_y_axis=False,
                budget=None,
                out_dir=".",
                ncol=2,
                x_label=None,
                y_label=None,
                show_doe=False,
                expert_configuration=None,
            )           

        # TODO add more plotting 

if __name__ == "__main__":
    main()
