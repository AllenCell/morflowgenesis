import asyncio

from hydra._internal.utils import _locate
from prefect.deployments import run_deployment
from slugify import slugify


async def run_step(step_cfg, prev_output):
    results = []

    step_fn = step_cfg["function"]
    step_type = step_cfg.get("step_type", "list")
    step_args = step_cfg["args"]
    step_args["step_name"] = step_fn.split(".")[-1]

    if step_type == "gather":
        step = _locate(step_fn)
        # return await run_deployment(f"{flow_name}/{deployment_name}", parameters=)
        return step(prev_output, **step_args)
    for datum in prev_output:
        payload = {"image_objects": datum, **step_args}
        flow_name = slugify(payload["step_name"])
        deployment_name = slugify(step_cfg.get("deployment_name", "default"))
        results.append(
            run_deployment(f"{flow_name}/{deployment_name}", parameters=payload, timeout=0)
        )
    results = await asyncio.gather(*results)

    return results
