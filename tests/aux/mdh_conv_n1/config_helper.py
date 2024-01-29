#!/usr/bin/python

#replaces Matrix size placeholders with actual values
import os



def replacePlaceholders(path: str, newPath, N: int, P: int, Q: int, K: int, C:int, R:int, S:int, optimization_iterations: int, slurmArrayTaskId: str, time_budget:int) -> None:
    with open(path, "r") as file:
        text = file.read()
    text = text.replace("_N_", str(N))
    text = text.replace("_P_", str(K))
    text = text.replace("_Q_", str(P))
    text = text.replace("_K_", str(Q))
    text = text.replace("_C_", str(C))
    text = text.replace("_R_", str(R))
    text = text.replace("_S_", str(S))
    text = text.replace("_iter_", str(optimization_iterations))
    text = text.replace("slurmArrayTaskId", slurmArrayTaskId)
    text = text.replace("timeBudgetVar", str(time_budget))
    with open(newPath, "w") as newFile:
        newFile.write(text)

path = os.path.dirname(__file__)
templates = ["/mdh_conv_n1_opentuner.json_template", 
            "/mdh_conv_n1_opentuner_biased.json_template",
            "/mdh_conv_n1_embedding_random_sampling.json_template",
            "/mdh_conv_n1_embedding_random_sampling_biased.json_template"]
json_files= ["/mdh_conv_n1_opentuner.json",
            "/mdh_conv_n1_opentuner_biased.json",
            "/mdh_conv_n1_embedding_random_sampling.json",
            "/mdh_conv_n1_embedding_random_sampling_biased.json"]

def createConfigFilesMdh(N: int, K: int, P: int, Q: int, C:int, R:int, S:int, optimization_iterations: int, slurmArrayTaskId: str, time_budget:int):
    for template, json in zip(templates, json_files):
        replacePlaceholders(path + template, path + json, N, K, P, Q, C, R, S, optimization_iterations, slurmArrayTaskId, time_budget)

def removeConfigFilesMdh():
    for json in json_files:
        if os.path.exists(path + json):
            os.remove(path + json)
