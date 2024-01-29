#!/usr/bin/python

#replaces Matrix size placeholders with actual values
# Ä = M, ä = m - 1
# Ö = N, ö = n - 1
# Ü = K, ü = k - 1
# ß = optimization_iterations

import os



def replacePlaceholders(path: str, newPath, M: int, N: int, K: int, optimization_iterations: int, slurmArrayTaskId: str, time_budget:int) -> None:
    with open(path, "r") as file:
        text = file.read()
    text = text.replace("Ä", str(M))
    text = text.replace("Ö", str(N))
    text = text.replace("Ü", str(K))
    text = text.replace("ß", str(optimization_iterations))
    text = text.replace("slurmArrayTaskId", slurmArrayTaskId)
    text = text.replace("timeBudgetVar", str(time_budget))
    with open(newPath, "w") as newFile:
        newFile.write(text)

path = os.path.dirname(__file__)
templates = ["/mdh_matmul_m1_opentuner.json_template", 
            "/mdh_matmul_m1_opentuner_biased.json_template",
            "/mdh_matmul_m1_embedding_random_sampling.json_template",
            "/mdh_matmul_m1_embedding_random_sampling_biased.json_template"]
json_files= ["/mdh_matmul_m1_opentuner.json",
            "/mdh_matmul_m1_opentuner_biased.json",
            "/mdh_matmul_m1_embedding_random_sampling.json",
            "/mdh_matmul_m1_embedding_random_sampling_biased.json"]

def createConfigFilesMdh(M: int, N: int, K: int, optimization_iterations: int, slurmArrayTaskId: str, time_budget:int):
    for template, json in zip(templates, json_files):
        replacePlaceholders(path + template, path + json, M, N, K, optimization_iterations, slurmArrayTaskId, time_budget)

def removeConfigFilesMdh():
    for json in json_files:
        if os.path.exists(path + json):
            os.remove(path + json)

