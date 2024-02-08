import asyncio
import logging
from pathlib import Path
import subprocess 
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from prefect import flow
from prefect.task_runners import SequentialTaskRunner
from prefect.server.schemas.states import StateType

from morflowgenesis.bin.run_step import run_step

# suppress info logging from dask
logging.getLogger("distributed").setLevel(logging.ERROR)


def save_workflow_config(working_dir, cfg):
    with open(Path(working_dir) / "workflow_config.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f)

def check_state(state, step_cfg):
    if state != StateType.COMPLETED:
        raise RuntimeError(f"Step {step_cfg['function']} completed with state {state}")

def setup_task_limits(step_cfg):
    """Set up task limits for the step"""
    task_limit = step_cfg.get('task_runner', {}).get('task_limit')
    if task_limit:
        # create unique tag based on task function and submission time
        task_name = f'{step_cfg["function"]}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
        # set task limits
        command = ["prefect", "concurrency-limit", "create", task_name, str(task_limit)]
        # Run the command
        subprocess.run(command, check=True)
        step_cfg['tags'] = [task_name]
        del step_cfg['task_runner']['task_limit']
    return step_cfg

def tear_down_task_limits(step_cfg):
    if 'tags' not in step_cfg:
        return
    command = ["prefect", "concurrency-limit", "delete", step_cfg['tags'][0]]
    # Run the command
    subprocess.run(command, check=True)


@flow(log_prints=True, task_runner=SequentialTaskRunner())
async def morflowgenesis(cfg):
    """Sequentially run config-specified steps, starting with the previous workflow state and
    passing output from step n-1 as input to step n."""
    working_dir = Path(cfg["working_dir"])
    working_dir.mkdir(exist_ok=True, parents=True)
    save_workflow_config(working_dir, cfg)

    for step_cfg in cfg["steps"]:
        step_cfg = setup_task_limits(step_cfg)
        result = run_step(step_cfg, working_dir / "_ImageObjectStore")
        tear_down_task_limits(step_cfg)
        check_state(result, step_cfg)

# default config is morflowgenesis/configs/workflow_config.yaml
@hydra.main(version_base="1.3", config_path="../configs/", config_name="workflow_config.yaml")
def main(_cfg: DictConfig):
    cfg = OmegaConf.to_container(_cfg, resolve=True)
    asyncio.run(morflowgenesis(cfg))


if __name__ == "__main__":
    main()
