import asyncio
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from prefect import flow
from prefect.task_runners import SequentialTaskRunner

from morflowgenesis.bin.run_step import run_step

# suppress info logging from dask
logging.getLogger("distributed").setLevel(logging.ERROR)


def save_workflow_config(working_dir, cfg):
    with open(Path(working_dir) / "workflow_config.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f)

@flow(log_prints=True, task_runner=SequentialTaskRunner())
async def morflowgenesis(cfg):
    """Sequentially run config-specified steps, starting with the previous workflow state and
    passing output from step n-1 as input to step n."""
    working_dir = Path(cfg["working_dir"])
    working_dir.mkdir(exist_ok=True, parents=True)
    save_workflow_config(working_dir, cfg)

    for step_cfg in cfg["steps"]:
        step_cfg.update({"deployment_name": cfg.get("deployment_name", "default")})
        run_step(step_cfg, working_dir / "_ImageObjectStore")

# default config is morflowgenesis/configs/workflow_config.yaml
@hydra.main(version_base="1.3", config_path="../configs/", config_name="workflow_config.yaml")
def main(_cfg: DictConfig):
    cfg = OmegaConf.to_container(_cfg, resolve=True)
    asyncio.run(morflowgenesis(cfg))


if __name__ == "__main__":
    main()
