from hydra._internal.utils import _locate
from morflowgenesis.utils import BlockDeployment, encode_dict_to_json_base64


def merge_dicts(dict1, dict2):
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1 and isinstance(dict1[key], dict):
            merge_dicts(dict1[key], value)
        else:
            dict1[key] = value
    return dict1


def deploy_step(cfg, step_cfg):
    step_fn = _locate(step_cfg["function"])
    # path to code within repo
    *entrypoint, flow_name = step_cfg["function"].split(".")
    entrypoint = "/".join(entrypoint) + f".py:{flow_name}"
    infra_overrides = merge_dicts(cfg['infra_overrides'], step_cfg.get('infra_overrides', {}))

    # we treate cpu steps as the default - null must be passed for steps that don't want to use GPU cluster
    if step_cfg.get("dask_cluster", True) is not None:
        merged_cluster_args= merge_dicts(cfg['dask_cluster'], step_cfg.get('dask_cluster', {}))
        encoded_dask_cluster = encode_dict_to_json_base64(merged_cluster_args)
        if "env" not in infra_overrides:
            infra_overrides["env"] = {}
        infra_overrides["env"]["DASK_CLUSTER"] = encoded_dask_cluster
    dep = BlockDeployment.build_from_flow(
        # build from flow args
        step_fn,
        step_cfg.get("deployment_name", "default"),
        apply=False,
        # Deployment kwargs
        path=cfg["path"],
        entrypoint=entrypoint,
        infra_overrides=infra_overrides,
        work_pool_name=cfg.get("work_pool_name"),
    )
    dep.apply(pull_cfg=cfg.get("pull"))
    return dep
