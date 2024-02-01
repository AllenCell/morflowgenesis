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
docker run -d --name prefect-postgres -v prefectdb:/var/lib/postgresql/data -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=yourTopSecretPassword -e POSTGRES_DB=prefect postgres:latest postgres -N 100
```

2. Set up a prefect server

```
prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://postgres:yourTopSecretPassword@localhost:5432/prefect"
prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
prefect server start
```

3. \[OPTIONAL\] For workflows generating LOTS of tasks (e.g. 1000s), you may need to increase the number of connections in the postgres database.

```
export PREFECT_SQLALCHEMY_POOL_SIZE=10
export PREFECT_SQLALCHEMY_MAX_OVERFLOW=100
```

## Running Example Workflows

1. Create Personal Access Token on GitHub with access to morflowgenesis repo (if one doesn't exist)
2. Create a [Secret block](https://prefect.a100.int.allencell.org/blocks/catalog/secret) that has the value of your PAT and copy the name of the block into your config under the `pull:secret_block_name` key. Under `pull`, also specify the name of the branch that you want to pull from.

```
prefect config set PREFECT_API_DATABASE_CONNECTION_URL="https://prefect.a100.int.allencell.org/api"

python morflowgenesis/bin/run_workflow.py --config-name nucleolus_morph.yaml
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

### Developer installation

pip install -e .\[dev\]

***Free software: Allen Institute Software License***
