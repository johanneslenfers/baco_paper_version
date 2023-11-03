from typing import Any, Dict, List, Union

import numexpr as ne
import numpy as np
import sys

from baco.param.parameters import Parameter

# TODO
# this is partially hard-coded for now 
def evaluate_constraints(
        constraints: List[str],
        configurations: Dict[str, List[Any]],
) -> List[bool]:
    """
    Checks configuration feasibility
    Input:
        - constraints: constraints to evaluate
        - configurations: configurations to evaluate ("original" representation)
    Returns:
        - List of booleans denoting whether the configurations are feasible

    Note that this requires some preprocessing - especially, the configurations should be provided as dicts.
    This preprocessing is performed by evaluate() in space.py.
    """

    print("evaluate constraints")

    # protect against empty configuration dicts
    if len(list(configurations.values())[0]) == 0:
        return []
    else:
        n_configurations = len(list(configurations.values())[0])

    # transform all permutation variables
    permutation_configurations = {}
    selection_configurations = {}

    for varname in configurations:

        # preprocess selection parameter 
        if varname == 'rewriting':
        # if type(configurations[varname][0][0]) == str:

            # varname_0 - give me value at index 0, result is value A
            for i in range(len(configurations[varname][-1])):
                selection_configurations[f"{varname}_{i}"] = [configurations[varname][j][i] if len(configurations[varname][j]) > i else '' for j in range(n_configurations)]

            # # varname_iA - give me index of A, result is index 0
            # # TODO: used possible parameter values 
            # # TODO: Evaluation will fail for this! 
            parameter_values: List[str] = ["A", "B", "C"]
            for parameter_value in parameter_values:
                selection_configurations[f"{varname}_i{parameter_value}"] = [tuple([index for index, value in enumerate(list(configurations[varname][j])) if value == parameter_value]) for j in range(n_configurations)]

            # varname_#A - amount of A, result is number 
            # TODO: used possible parameter values 
            parameter_values: List[str] = ["A", "B", "C"]
            for parameter_value in parameter_values:
                selection_configurations[f"{varname}_N{parameter_value}"] = [configurations[varname][j].count(parameter_value) for j in range(n_configurations)]
                
        # preprocess permutation parameter 
        elif type(configurations[varname][0]) in (list, tuple):
            for i in range(len(configurations[varname][0])):
                permutation_configurations[f"{varname}_{i}"] = [configurations[varname][j][i] for j in range(n_configurations)]
                permutation_configurations[f"{varname}_i{i}"] = [list(configurations[varname][j]).index(i) for j in range(n_configurations)]


    feasible = np.array([True for x in range(n_configurations)])
    feasible2 = [True for _ in range(n_configurations)]

    for constraint in constraints:

        # TODO: check if we have selection constraint 
        # evaluate selection constraints separately 
        # print(f"constraint: {constraint}")
        if "rewriting_0" in constraint:
            # print("\n")
            # print(f"Selection Constraint evaluation: {constraint}")

            # get keyword from constraint 
            import re
            index = re.search(r'[<>=!]', constraint).start()
            keyword = constraint[:index]
            result = re.split(r'==|=|<|>|!=', constraint)[-1]

            # print(f"keyword: {keyword}")
            # print(f"result: {result}")

            # TODO speed this up 
            for varname in configurations:
                feasible_result = [configuration == result for configuration in selection_configurations[keyword]]
                # print(f"feasible_configuration: {feasible_result}")

                feasible2 = [a and b for a,b in zip(feasible2, feasible_result)]


            # print(f"feasile2: {feasible2}")

            feasible = feasible & np.array(feasible2)

            # constraints that compare against string 
            # constraints that compare against multiple indices 
            # Cardinality constraint can be evaluated in baco

        elif "rewriting_i" in constraint:
            # this currently checks for specific configuration
            # TODO change this to "or" 

            # Currently
            # rewriting_iA=0,1 -> index of A has to be 0 and 1 
            # Goal
            # rewriting_iA=0,1 -> index of A has to be 0 or 1 or both 

            # parse constraint 
            # get keyword from constraint 
            import re
            index = re.search(r'[<>=!]', constraint).start()
            keyword = constraint[:index]
            result = re.split(r'==|=|<|>|!=', constraint)[-1]

            # get right side of contraint 
            # give me the index of A
            value = tuple([int(num) for num in result.split(',')])

            feasible3 = [all(elem in configuration for elem in value) for configuration in selection_configurations[keyword]]
            feasible = feasible & np.array(feasible3)

            pass 

        # evalute baco constraints (num_expr)
        else:
            feasible = feasible & ne.evaluate(constraint, {**configurations, **permutation_configurations, **selection_configurations})
    

    return list(feasible)

def filter_conditional_values(
        parameter: Parameter,
        constraints: Union[List[str], None],
        partial_configuration: Dict[str, Any]
) -> List[Any]:
    """
    Returns all of its values which are feasible with regards to its constraints given previous values given in partial_configuration.
        Input:
            - parameter: parameter to find feasible values for
            - constraints: constraints to evaluate
            - partial_configuration: configuration so far ("original" representation)
        Returns:
            - List of feasible parameter values ("internal" representation)
    """
    if constraints is None:
        return parameter.values
    configurations = {
        **{kv[0]: [kv[1]] * len(parameter.values) for kv in partial_configuration.items()},
        **{parameter.name: [parameter.convert(v, "internal", "original") for v in parameter.values]}
    }
    feasible = evaluate_constraints(constraints, configurations)
    return [value for idx, value in enumerate(parameter.values) if feasible[idx]]
