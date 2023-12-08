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

pip install -e .[dev]

***Free software: Allen Institute Software License***
