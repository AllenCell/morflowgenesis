import asyncio

import hydra
from hydra._internal.utils import _locate
from prefect import flow
from prefect.deployments.deployments import run_deployment


async def run_step(step_cfg, prev_output):
    results = []

    step_fn = step_cfg["function"]
    step_type = step_cfg["step_type"]
    step_args = step_cfg["args"]

    if step_type == "gather":
        step = _locate(step_fn)
        return step(prev_output, **step_args)

    for datum in prev_output:
        payload = {"image_objects": datum, **step_args}
        flow_name = payload["step_name"]
        deployment_name = step_cfg.get("deployment_name", "default")

        results.append(
            run_deployment(f"{flow_name}/{deployment_name}",
                           parameters=payload, timeout=0)
        )

    results = await asyncio.gather(*results)

    return results
