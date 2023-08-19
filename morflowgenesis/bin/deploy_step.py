from copy import copy

from hydra._internal.utils import _locate
from hydra.utils import instantiate
from prefect import flow
from prefect.deployments import build_from_flow, run_deployment

from morflowgenesis.utils import flatten_dict


def _merge_configs(cfg, step_cfg, key):
    new = copy(cfg[key])
    new.update(step_cfg.get(key, {}))
    return new


def deploy_step(cfg, step_cfg):
    step_fn = _locate(step_cfg["function"])
    task_runner_cfg = _merge_configs(cfg, step_cfg, "task_runner")
    task_runner = instantiate(task_runner_cfg)
    step_flow = flow(step_fn, task_runner=task_runner)

    *entrypoint, flow_name = step_cfg["function"].split(".")
    entrypoint = "/".join(entrypoint.split(".")) + f".py:{flow_name}"

    infra_overrides = _merge_configs(cfg, step_cfg, "infra_overrides")

    return build_from_flow(
        step_flow,
        step_cfg.get("deployment_name", "default"),
        apply=True,
        storage=cfg.storage,
        path=cfg.path,
        entrypoint=entrypoint,
        infra_overrides=flatten_dict(infra_overrides),
    )
