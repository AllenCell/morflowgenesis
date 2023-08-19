import asyncio
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from prefect import flow
from prefect.deployments import build_from_flow, run_deployment

from morflowgenesis.steps.load_workflow_state import get_workflow_state

from .deploy_step import deploy_step
from .run_step import run_step


def save_workflow_config(working_dir, cfg):
    with open(working_dir / "workflow_config.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f)


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
async def main(cfg: DictConfig):
    deployment_name = cfg.get("deployment_name").get("default")
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)

    if cfg.deploy:
        await build_from_flow(
            morflowgenesis,
            deployment_name,
            apply=True,
            storage=cfg.storage,
            path=cfg.path,
            entrypoint=cfg.entrypoint,
            infra_overrides=cfg.infra_overrides,
        )

        deployments = []
        for step_cfg in cfg["steps"]:
            deployments.append(deploy_step(cfg, step_cfg))
        await asyncio.gather(*deployments)

    await run_deployment(
        name=f"morflowgenesis/{deployment_name}", parameters=resolved_cfg, timeout=0
    )


if __name__ == "__main__":
    main()
