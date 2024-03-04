# morflowgenesis

[![Build Status](https://github.com/AllenCell/morflowgenesis/workflows/Build%20Master/badge.svg)](https://github.com/AllenCell/morflowgenesis/actions)
[![Code Coverage](https://codecov.io/gh/AllenCell/morflowgenesis/branch/master/graph/badge.svg)](https://codecov.io/gh/AllenCell/morflowgenesis)

Morflowgenesis is a python package for creating modular and configurable workflows for morphogenesis-style projects and validation using prefect and hydra.

______________________________________________________________________

## Installation

```
python3 -m venv morflowgenesis
source morflowgenesis/bin/activate

git clone  https://github.com/AllenCell/morflowgenesis.git
cd morflowgenesis
pip install -r requirements.txt
pip install -e .
```

## Setup

1. Set up a postgres database for tracking workflow artifacts

```
docker run -d --name prefect-postgres -v prefectdb:/var/lib/postgresql/data -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=yourTopSecretPassword -e POSTGRES_DB=prefect --shm-size=1g postgres:latest -c 'max_connections=800' -c 'shared_buffers=1000MB'
```

2. Set up a prefect server

```
prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://postgres:yourTopSecretPassword@localhost:5432/prefect"
prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
prefect server start

[OPTIONAL]
nohup prefect server start > output.log &
```

3. \[OPTIONAL\] For workflows generating LOTS of tasks (e.g. 1000s), you may need to increase the number of connections in the postgres database.

```
export PREFECT_SQLALCHEMY_POOL_SIZE=100
export PREFECT_SQLALCHEMY_MAX_OVERFLOW=-1
```

4. \[OPTIONAL\] Add task runners
   By default, tasks are run using `prefect.task_runners.SequntialTaskRunner`, which implements a simple parallelization strategy. If you want to run tasks differently, you can pass a `task-runner` argument to any step. For example,

```
- function: morflowgenesis.steps.foo
    args:
        foo: bar
    task_runner:
        task_limit: 10
        _target_: prefect_dask.DaskTaskRunner
        cluster_class: distributed.LocalCluster
        cluster_kwargs:
            n_workers: ${..task_limit}
            memory_limit: 5Gi
            processes: True
            threads_per_worker: 4
            resources:
            cpu: 1
```

will create a `DaskTaskRunner` with a `LocalCluster` and a memory limit of 5Gi per worker. Please see the [prefect task runner docs](https://docs.prefect.io/latest/concepts/task-runners/) for more information on available task runners.

Under `task_runner`, you can include a `_target_` with the task runner you want to use and a `task_limit` which sets a unique `prefect concurrency-limit` for your step. Limiting task concurrency can be useful for resource-intensive steps.
If runs are cancelled before their concurrency limits can be cleaned up, you may need to manually clean up the unused concurrency limits. You can do this with the following command (WARNING this will delete the first 200 concurrency limits)

```
prefect concurrency-limit ls --limit 200 | grep morflowgenesis | awk '{print $2}' | while read arg; do prefect concurrency-limit delete $arg; done
```

## Running Workflows

morflowgenesis is designed to run `workflows`. To run the `nucmorph.yaml` workflow from the `configs/workflow` directory we can do:

```
python morflowgenesis/bin/run_workflow.py workflow=nucmorph.yaml
```

The order of steps in this workflow are determined by the order of steps imported under the `defaults` key in the `workflow` config. This will import the default config for each step as the base of the workflow. Defaults are defined in the `configs/steps` directory. Most of our workflows require modifications of the defaults (for example specifying input and output step names). We can override default arguments under the `steps` key in the `workflow` config. A `paths` config is also created that contains useful paths within your project. For example, `${paths.model_dir}` points to the `configs/model` directory, where you can store model-related configs. For running the same workflow on multiple inputs, you can use the `configs/params` folder to define a different parameter set for a multirun. If you use this, makesure the parameters you sweep over in the multirun are referenced via `params.<param_name>` in your workflow config. To use the params in a multirun:

```
python morflowgenesis/bin/run_workflow.py workflow=nucmorph.yaml params=param_folder1.yaml,param_folder2.yaml -m
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

### Developer installation

pip install -e .\[dev\]

***Free software: Allen Institute Software License***
