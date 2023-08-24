import os

import pandas as pd
from aicsimageio import AICSImage
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect_dask import DaskTaskRunner

from morflowgenesis.utils.image_object import ImageObject, StepOutput

DASK_ADDRESS = os.environ.get("DASK_ADDRESS", None)


@task
def generate_object(
    row, working_dir, step_name, source_column, non_source_columns, metadata_column=None
):
    source_img = AICSImage(row[source_column])
    # add metadata
    metadata = {"S": source_img.scenes, "T": source_img.dims.T - 1, "C": source_img.dims.C - 1}
    if metadata_column is not None:
        metadata.update(row.get(metadata_column))

    obj = ImageObject(working_dir, row[source_column], metadata)
    for col in [source_column] + non_source_columns:
        step_output = StepOutput(
            working_dir, step_name, col, "image", image_id=obj.id, path=row[col]
        )
        obj.add_step_output(step_output)
    obj.save()
    return obj


@flow(task_runner=DaskTaskRunner(address=DASK_ADDRESS) if DASK_ADDRESS else SequentialTaskRunner())
def generate_objects(
    image_objects,
    working_dir,
    step_name,
    csv_path,
    source_column,
    non_source_columns,
    metadata_column=None,
):
    """Generate a new image object for each row in the csv file."""
    df = pd.read_csv(csv_path)

    already_run = [im_obj.source_path for im_obj in image_objects]
    new_image_objects = []
    for row in df.itertuples():
        row = row._asdict()
        if row[source_column] not in already_run:
            new_image_objects.append(
                generate_object.submit(
                    row, working_dir, step_name, source_column, non_source_columns, metadata_column
                )
            )
    new_image_objects = [im_obj.result() for im_obj in new_image_objects]
    return image_objects + new_image_objects
