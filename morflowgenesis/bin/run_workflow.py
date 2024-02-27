import asyncio
import logging
import os
from pathlib import Path

import hydra
import pyrootutils
from omegaconf import DictConfig, OmegaConf
from prefect import flow
from prefect.task_runners import SequentialTaskRunner

from morflowgenesis.bin.run_step import run_step
from morflowgenesis.utils.rich_utils import print_config_tree

# suppress info logging from dask
logging.getLogger("distributed").setLevel(logging.ERROR)


@flow(log_prints=True, task_runner=SequentialTaskRunner())
async def morflowgenesis(cfg):
    """Sequentially run config-specified steps."""
    workflow = cfg["workflow"]
    working_dir = Path(workflow["working_dir"])
    working_dir.mkdir(exist_ok=True, parents=True)

    for step_name, step_cfg in workflow["steps"].items():
        await run_step(step_cfg, working_dir / "_ImageObjectStore")


def clean_config(cfg):
    cfg["workflow"]["steps"] = {
        k: v for k, v in cfg["workflow"]["steps"].items() if "function" in v.keys()
    }
    return cfg


@hydra.main(version_base="1.3", config_path="../configs", config_name="workflow.yaml")
def main(_cfg: DictConfig):
    cfg = OmegaConf.to_container(_cfg, resolve=True)
    cfg = clean_config(cfg)
    print_config_tree(cfg, resolve=True, save_to_file=True)
    asyncio.run(morflowgenesis(cfg))


if __name__ == "__main__":
    main()
