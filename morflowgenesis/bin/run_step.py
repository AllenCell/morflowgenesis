import asyncio

from hydra._internal.utils import _locate
from prefect import flow, task
from prefect.deployments import run_deployment
from prefect.task_runners import ConcurrentTaskRunner
from slugify import slugify

from morflowgenesis.utils import ImageObject


@task()
def _is_run(path, step_name, output_name):
    image_object = ImageObject.parse_file(path)
    # check if step already run
    if image_object.step_is_run(f"{step_name}_{output_name}"):
        print(f"Skipping step {step_name}_{output_name} for image {image_object.id}")
        return
    return path


@flow(task_runner=ConcurrentTaskRunner, log_prints=True)
def get_objects_to_run(working_dir, step_name, output_name):
    objects_to_run = []
    for object_path in working_dir.glob("*.json"):
        objects_to_run.append(_is_run.submit(object_path, step_name, output_name))
    objects_to_run = [obj.result() for obj in objects_to_run]
    objects_to_run = [obj for obj in objects_to_run if obj is not None]
    return objects_to_run


async def run_step(step_cfg, object_store_path):
    step_fn = step_cfg["function"]
    step_type = step_cfg.get("step_type", "gather")
    step_args = step_cfg["args"]
    step_name = step_fn.split(".")[-1]

    flow_name = slugify(step_name)
    deployment_name = slugify(step_cfg.get("deployment_name", "default"))
    full_deployment_name = f"{flow_name}/{deployment_name}"
    if step_type == "gather":
        out = await run_deployment(
            full_deployment_name,
            parameters=step_args,
        )
        return [out]
    else:
        # checking which objects to run here prevents overhead on the cluster and excess job creation.
        objects_to_run = get_objects_to_run(
            object_store_path, step_name, step_args["output_name"]
        )
        out = await asyncio.gather(
            *[
                run_deployment(
                    full_deployment_name,
                    parameters={"image_object_path": object_path, **step_args},
                )
                for object_path in objects_to_run
            ]
        )
        return out


def run_step_local(step_cfg, object_store_path):
    step_fn = step_cfg["function"]
    step_type = step_cfg.get("step_type", "gather")
    step_args = step_cfg["args"]
    step_name = step_fn.split(".")[-1]
    step_fn = _locate(step_fn)

    if step_type == "init":
        step_fn(**step_args)
    else:
        # checking which objects to run here prevents overhead on the cluster and excess job creation.
        objects_to_run = get_objects_to_run(
            object_store_path, step_name, step_args.get("output_name")
        )
        if len(objects_to_run) == 0:
            return
        if step_type == "gather":
            step_args.update({"image_object_paths": objects_to_run})
            step_fn(**step_args)
        else:
            for object_path in objects_to_run:
                step_args.update({"image_object_paths": [object_path]})
                step_fn(**step_args)
