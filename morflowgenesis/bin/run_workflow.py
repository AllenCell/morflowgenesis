import asyncio
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from prefect import flow

# from prefect.blocks.core import Block
from prefect.deployments import run_deployment
from prefect.task_runners import SequentialTaskRunner

from morflowgenesis.bin.deploy_step import deploy_step
from morflowgenesis.bin.run_step import run_step
from morflowgenesis.steps.load_workflow_state import get_workflow_state
from morflowgenesis.utils import BlockDeployment


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

    out = get_workflow_state(cfg)
    for step_cfg in cfg["steps"]:
        out = await run_step(step_cfg, out)


# default config is morflowgenesis/configs/workflow/config.yaml
@hydra.main(version_base="1.3", config_path="../configs/workflow", config_name="config.yaml")
def main(_cfg: DictConfig):
    cfg = OmegaConf.to_container(_cfg, resolve=True)
    deployment_name = cfg.get("deployment_name", "default")

    debug = cfg.get("debug", False)

    if cfg.get("deploy", False):
        if not debug:
            dep = BlockDeployment.build_from_flow(
                morflowgenesis,
                deployment_name,
                apply=False,
                path=cfg["path"],
                entrypoint=cfg["entrypoint"],
                infra_overrides=cfg["infra_overrides"],
            )
            dep.apply(cfg["pull"])

        for step_cfg in cfg["steps"]:
            deploy_step(cfg, step_cfg)

    if not debug:
        run_deployment(name=f"morflowgenesis/{deployment_name}", parameters=cfg)
    else:
        # run superworkflow locally, better error readouts
        asyncio.run(morflowgenesis(cfg))


if __name__ == "__main__":
    # asyncio.run(main())
    main()
