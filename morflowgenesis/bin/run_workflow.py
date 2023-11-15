import asyncio
from copy import deepcopy
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from prefect import flow
from prefect.deployments import run_deployment
from prefect.server.schemas.states import StateType
from prefect.task_runners import SequentialTaskRunner
from slugify import slugify

from morflowgenesis.bin.deploy_step import deploy_step
from morflowgenesis.bin.run_step import run_step, run_step_local
from morflowgenesis.utils import BlockDeployment


def save_workflow_config(working_dir, cfg):
    with open(Path(working_dir) / "workflow_config.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f)


def check_state(state_list, step_cfg):
    for r in state_list:
        if r.state.type != StateType.COMPLETED:
            raise RuntimeError(f"Step {step_cfg['function']} completed with state {r.state_name}")


@flow(log_prints=True, task_runner=SequentialTaskRunner())
async def morflowgenesis(cfg):
    """Sequentially run config-specified steps, starting with the previous workflow state and
    passing output from step n-1 as input to step n."""
    working_dir = Path(cfg["working_dir"])
    working_dir.mkdir(exist_ok=True, parents=True)
    save_workflow_config(working_dir, cfg)

    for step_cfg in cfg["steps"]:
        step_cfg.update({"deployment_name": cfg.get("deployment_name", "default")})
        if cfg.get("local_run", False):
            run_step_local(step_cfg, working_dir / "_ImageObjectStore")
        else:
            result = await run_step(step_cfg, working_dir / "_ImageObjectStore")
            check_state(result, step_cfg)


# default config is morflowgenesis/configs/workflow_config.yaml
@hydra.main(version_base="1.3", config_path="../configs/", config_name="workflow_config.yaml")
def main(_cfg: DictConfig):
    cfg = OmegaConf.to_container(_cfg, resolve=True)

    if cfg.get("local_run", False):
        asyncio.run(morflowgenesis(cfg))
    else:
        deployment_name = slugify(cfg.get("deployment_name", "default"))
        dep = BlockDeployment.build_from_flow(
            morflowgenesis,
            deployment_name,
            apply=False,
            path=cfg["path"],
            entrypoint=cfg["entrypoint"],
            infra_overrides=cfg["infra_overrides"],
            work_pool_name=cfg.get("work_pool_name"),
        )
        dep.apply(cfg["pull"])

        for step_cfg in cfg["steps"]:
            deploy_step(deepcopy(cfg), step_cfg)

        asyncio.run(
            run_deployment(name=f"morflowgenesis/{deployment_name}", parameters={"cfg": cfg})
        )


if __name__ == "__main__":
    main()
