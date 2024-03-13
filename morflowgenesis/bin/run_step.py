from datetime import datetime

from hydra._internal.utils import _locate
from hydra.utils import instantiate
from prefect import flow
from prefect.client.orchestration import get_client
from prefect.exceptions import ObjectNotFound
from prefect.server.schemas.states import StateType
from prefect.task_runners import ConcurrentTaskRunner

from morflowgenesis.utils import ImageObject, run_flow


def _is_run(path, step_name, output_name):
    image_object = ImageObject.parse_file(path)
    # check if step already run
    if image_object.step_is_run(f"{step_name}/{output_name}"):
        print(f"Skipping {step_name}/{output_name} for  image {image_object.id}")
        return
    return image_object


@flow(log_prints=True)
def get_objects_to_run(object_store_path, step_name, output_name, limit=-1):
    fns = list(object_store_path.glob("*.json"))
    if limit > 0:
        fns = fns[:limit]
    objects_to_run = [_is_run(object_path, step_name, output_name) for object_path in fns]
    objects_to_run = [obj for obj in objects_to_run if obj is not None]
    print(f"Running {len(objects_to_run)} objects for {step_name}/{output_name}")
    return objects_to_run


def check_state(state, step_name):
    if state != StateType.COMPLETED:
        raise RuntimeError(f"Step {step_name} completed with state {state}")


async def setup_task_limits(step_cfg, step_name):
    """Set prefect concurrency-limit based on step configuration."""
    task_limit = step_cfg.get("task_runner", {}).get("task_limit")
    step_cfg["tags"] = []
    if task_limit:
        # create unique tag based on task function and submission time
        task_name = f'{step_name}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
        # set task limits
        async with get_client() as client:
            await client.create_concurrency_limit(tag=task_name, concurrency_limit=task_limit)
        print(f"Set task limit for {step_name} to {task_limit}")
        del step_cfg["task_runner"]["task_limit"]
        step_cfg["task_runner"].pop("memory_limit", None)

        step_cfg["tags"] = [task_name]

    return step_cfg


async def tear_down_task_limits(step_cfg):
    """Delete prefect concurrency-limit based on step configuration."""
    if len(step_cfg.get("tags", [])) == 0:
        return
    async with get_client() as client:
        try:
            tag = step_cfg["tags"][0]
            await client.delete_concurrency_limit_by_tag(tag=tag)
            print(f"Deleted task limit for tag {tag}")
        except ObjectNotFound:
            print("Concurrency limit not found")


async def run_step(step_cfg, object_store_path, limit=-1):
    """Run a step in the pipeline."""
    # initialize function
    step_fn = step_cfg.pop("function")
    step_name = step_fn.split(".")[-1]
    step_fn = _locate(step_fn)

    # checking which objects to run here prevents overhead on the cluster and excess job creation.
    objects_to_run = get_objects_to_run(object_store_path, step_name, step_cfg.get("output_name"))
    if len(objects_to_run) == 0 and object_store_path.exists():
        return StateType.COMPLETED

    # set up task runner
    step_cfg = await setup_task_limits(step_cfg, step_name)
    task_runner = step_cfg.pop("task_runner", None)
    task_runner = instantiate(task_runner) if task_runner is not None else ConcurrentTaskRunner()

    step_cfg.update({"image_objects": objects_to_run})
    result = run_flow(step_fn, task_runner, **step_cfg)
    await tear_down_task_limits(step_cfg)
    check_state(result, step_name)
