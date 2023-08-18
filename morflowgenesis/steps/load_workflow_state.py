import pickle
from pathlib import Path

from prefect import flow, get_run_logger, task
from prefect.task_runners import ConcurrentTaskRunner


@task
def load_image_objects(obj_path):
    with open(obj_path, "rb") as f:
        return pickle.load(f)


@flow(task_runner=ConcurrentTaskRunner())
def get_workflow_state(cfg):
    """Load image objects as previous workflow state."""
    logger = get_run_logger()
    existing_objects = []
    img_object_dir = Path(cfg["working_dir"]) / "_ImageObjectStore"
    for obj_path in img_object_dir.glob("*.pkl"):
        existing_objects.append(load_image_objects.submit(obj_path))
    logger.info(f"Loaded {len(existing_objects)} image objects")
    return [x.result() for x in existing_objects]
