import asyncio

from prefect.deployments import run_deployment
from slugify import slugify
from prefect.states import Completed


async def run_step(step_cfg, prev_output):
    step_fn = step_cfg["function"]
    step_type = step_cfg.get("step_type", "list")
    step_args = step_cfg["args"]
    step_args["step_name"] = step_fn.split(".")[-1]
    flow_name = slugify(step_args["step_name"])
    deployment_name = slugify(step_cfg.get("deployment_name", "default"))
    full_deployment_name = f"{flow_name}/{deployment_name}"
    if step_type == "gather":
        out = await run_deployment(
            full_deployment_name,
            parameters=step_args,
        )
        if not out.state.type !=  Completed:
            raise RuntimeError(f"Step {step_fn} completed with state {out.state_name}")
    else:
        out = await asyncio.gather(
            *[run_deployment(
                    full_deployment_name, 
                    parameters = {"image_object_path": object_path, **step_args}
                ) 
                for object_path in prev_output.glob('*.json')
            ]
        )
        for result in out:
            if not result.state.type !=  Completed:
                raise RuntimeError(f"Step {step_fn} completed with state {result.state_name}")

