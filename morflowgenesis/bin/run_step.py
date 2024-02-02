from hydra._internal.utils import _locate
from hydra.utils import instantiate
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner

from morflowgenesis.utils import ImageObject, run_flow


@task()
def _is_run(path, step_name, output_name):
    image_object = ImageObject.parse_file(path)
    # check if step already run
    if image_object.step_is_run(f"{step_name}_{output_name}"):
        print(f"Skipping step {step_name}_{output_name} for image {image_object.id}")
        return
    return path


@flow(task_runner=ConcurrentTaskRunner, log_prints=True)
def get_objects_to_run(object_store_path, step_name, output_name):
    objects_to_run = []
    for object_path in object_store_path.glob("*.json"):
        objects_to_run.append(_is_run.submit(object_path, step_name, output_name))
    objects_to_run = [obj.result() for obj in objects_to_run]
    objects_to_run = [obj for obj in objects_to_run if obj is not None]
    return objects_to_run

def run_step(step_cfg, object_store_path):
    step_fn = step_cfg["function"]
    step_type = step_cfg.get("step_type", "gather")
    tags = step_cfg.get("tags", [])
    task_runner = step_cfg.get("task_runner")
    task_runner = instantiate(task_runner) if task_runner is not None else ConcurrentTaskRunner()
    
    step_args = step_cfg["args"]
    step_name = step_fn.split(".")[-1]
    step_fn = _locate(step_fn)

    # checking which objects to run here prevents overhead on the cluster and excess job creation.
    objects_to_run = get_objects_to_run(
        object_store_path, step_name, step_args.get("output_name")
    )
    if len(objects_to_run) == 0 and object_store_path.exists():
        return
    step_args.update({"image_object_paths": objects_to_run})
    run_type = "images" if step_type == "gather" else "objects" 
    run_flow(step_fn, task_runner, run_type, tags, **step_args)
