# morflowgenesis

[![Build Status](https://github.com/AllenCell/morflowgenesis/workflows/Build%20Master/badge.svg)](https://github.com/AllenCell/morflowgenesis/actions)
[![Code Coverage](https://codecov.io/gh/AllenCell/morflowgenesis/branch/master/graph/badge.svg)](https://codecov.io/gh/AllenCell/morflowgenesis)

Morflowgenesis is a python package for creating modular and configurable workflows for morphogenesis-style projects and validation using prefect and hydra.

______________________________________________________________________

## Installation

`python3 -m venv morflowgenesis`
`source morflowgenesis/bin/activate`

` git clone  https://github.com/AllenCell/morflowgenesis.git`
` cd morflowgenesis`
` pip install -r requirements.txt`
` pip install -e .`

## Running Example Workflows

### Setting up a prefect server

More details [here](https://docs.prefect.io/latest/host/)
`docker run -d --name prefect-postgres -v prefectdb:/var/lib/postgresql/data -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=yourTopSecretPassword -e POSTGRES_DB=prefect postgres:latest `
`prefect server start `

In a new terminal window:
`prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://postgres:yourTopSecretPassword@localhost:5432/prefect"`

`python morflowgenesis/bin/run_workflow.py --config-dir morflowgenesis/configs/workflow --config-name nucleolus_morph.yaml`

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

### Developer installation

`pip install -e .[dev]`

***Free software: Allen Institute Software License***
