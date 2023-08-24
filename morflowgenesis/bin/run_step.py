import asyncio

from prefect.deployments import run_deployment
from slugify import slugify


async def run_step(step_cfg, prev_output):
    results = []

    step_fn = step_cfg["function"]
    step_type = step_cfg.get("step_type", "list")
    step_args = step_cfg["args"]
    step_args["step_name"] = step_fn.split(".")[-1]

    flow_name = slugify(step_args["step_name"])
    deployment_name = slugify(step_cfg.get("deployment_name", "default"))
    full_deployment_name = f"{flow_name}/{deployment_name}"

    if step_type == "gather":
        payload = {"image_objects": prev_output, **step_args}
        out = await run_deployment(
            full_deployment_name,
            parameters=payload,
        )
        return out.state.result()
    for datum in prev_output:
        payload = {"image_objects": datum, **step_args}
        results.append(
            run_deployment(
                full_deployment_name,
                parameters=payload,
                timeout=0,
            )
        )
    # this isn't actually running stuff
    results = await asyncio.gather(*results)
    return [r.state.result() for r in results]
