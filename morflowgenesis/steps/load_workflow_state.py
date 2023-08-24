import os
from pathlib import Path

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect_dask import DaskTaskRunner

from morflowgenesis.utils import ImageObject

DASK_ADDRESS = os.environ.get("DASK_ADDRESS", None)


@task
def load_image_objects(obj_path):
    return ImageObject.parse_file(obj_path)


@flow(task_runner=DaskTaskRunner(address=DASK_ADDRESS) if DASK_ADDRESS else SequentialTaskRunner())
def get_workflow_state(cfg):
    """Load image objects as previous workflow state."""
    existing_objects = []
    img_object_dir = Path(cfg["working_dir"]) / "_ImageObjectStore"
    for obj_path in img_object_dir.glob("*.json"):
        existing_objects.append(load_image_objects.submit(obj_path))
    print(f"Loaded {len(existing_objects)} image objects")
    return [x.result() for x in existing_objects]
