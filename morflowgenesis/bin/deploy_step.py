from copy import copy

from hydra._internal.utils import _locate

from morflowgenesis.utils import BlockDeployment


def _merge_configs(cfg, step_cfg, key):
    new = copy(cfg[key])
    new.update(step_cfg.get(key, {}))
    return new


def deploy_step(cfg, step_cfg):
    step_fn = _locate(step_cfg["function"])
    # path to code within repo
    *entrypoint, flow_name = step_cfg["function"].split(".")
    entrypoint = "/".join(entrypoint) + f".py:{flow_name}"

    infra_overrides = _merge_configs(cfg, step_cfg, "infra_overrides")
    dep = BlockDeployment.build_from_flow(
        # build from flow args
        step_fn,
        step_cfg.get("deployment_name", "default"),
        apply=False,
        # Deployment kwargs
        # storage=Block.load(cfg["storage_block"]),
        path=cfg["path"],
        entrypoint=entrypoint,
        infra_overrides=infra_overrides,
        work_pool_name=cfg.get("work_pool_name"),
    )
    dep.apply(pull_cfg=cfg.get("pull"))
    return dep
