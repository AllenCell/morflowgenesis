import asyncio
import logging
import os
from pathlib import Path

import hydra
import pandas as pd
import tqdm
from omegaconf import DictConfig, OmegaConf
from prefect import flow
from prefect.task_runners import SequentialTaskRunner

from morflowgenesis.bin.run_step import run_step
from morflowgenesis.utils import ImageObject
from morflowgenesis.utils.rich_utils import print_config_tree

# suppress info logging from dask
logging.getLogger("distributed").setLevel(logging.ERROR)


def append(data, k, v):
    if k not in data:
        data[k] = []
    data[k].append(v)
    return data


def generate_summary(working_dir):
    """Generate a summary of the workflow results."""
    image_objects = [
        ImageObject.parse_file(fn)
        for fn in tqdm.tqdm(
            (working_dir / "_ImageObjectStore/").glob("*"), desc="Loading image objects"
        )
    ]
    data = {"id": [], "source_path": []}

    for obj in tqdm.tqdm(image_objects, desc="Creating CSV"):
        data["id"].append(obj.id)
        data["source_path"].append(obj.source_path)
        for k, v in obj.metadata.items():
            append(data, k, v)
        for step in obj.steps:
            append(data, step, obj.steps[step].path)
    pd.DataFrame(data).to_csv(working_dir / "metadata.csv", index=False)


@flow(log_prints=True, task_runner=SequentialTaskRunner())
async def morflowgenesis(cfg):
    """Sequentially run config-specified steps."""
    workflow = cfg["workflow"]
    working_dir = Path(workflow["working_dir"])
    working_dir.mkdir(exist_ok=True, parents=True)

    for step_name, step_cfg in workflow["steps"].items():
        await run_step(step_cfg, working_dir / "_ImageObjectStore")
    generate_summary(working_dir)


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
