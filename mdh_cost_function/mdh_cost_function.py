import json
import os.path
import subprocess
from tempfile import TemporaryDirectory
from typing import List, Dict, Union
from sys import maxsize


def mdh_cost_function(path: str, routine: str, input_size: List[int], warmups: int = 3, evaluations: int = 5):
    # prepare benchmark directory and command
    bench_directory = TemporaryDirectory()
    bench_command = [f'{path}/bench_md_hom_blocked_v3_{routine}',
                     f'--path {bench_directory.name}',
                     f'-c {bench_directory.name}/config.json',
                     f'-w {warmups}', f'-e {evaluations}']
    if routine == 'matmul':
        cc_dims = 2
        cb_dims = 1
    elif routine == 'conv':
        cc_dims = 4
        cb_dims = 3
    else:
        raise ValueError(f'unrecognized routine: {routine}')
    if len(input_size) != cc_dims + cb_dims:
        raise ValueError(f'incorrect number of input sizes for routine {routine}, expected {cc_dims + cb_dims}')
    for idx, size in enumerate(input_size):
        if idx < cc_dims:
            mdh_dim = 'l-' + str(idx + 1)
        else:
            mdh_dim = 'r-' + str(idx - cc_dims + 1)
        bench_command.append(f'--input-size-{mdh_dim} {size}')

    # define cost function
    global cost_function
    def cost_function(config: Dict[str, Union[int, float]]) -> Dict[str, Union[float, bool]]:
        config_path = f'{bench_directory.name}/config.json'
        runtime_path = f'{bench_directory.name}/runtimes_min'

        # remove old benchmarking results
        if os.path.exists(runtime_path):
            os.remove(runtime_path)

        # write current configuration to file
        with open(config_path, 'w') as config_file:
            json.dump(config, config_file)

        # benchmark configuration
        subprocess.run(bench_command, cwd=path)

        # get runtime, if valid configuration
        if os.path.exists(runtime_path):
            with open(runtime_path, 'r') as runtime_file:
                runtime = int(runtime_file.readline())
            return {'Valid': True, 'runtime': runtime}
        else:
            return {'runtime': maxsize, 'Valid': False}

    return cost_function
