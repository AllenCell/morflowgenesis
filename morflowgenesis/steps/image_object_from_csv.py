from pathlib import Path
import json
import pandas as pd
from aicsimageio import AICSImage
from prefect import flow, task

from morflowgenesis.utils.create_temporary_dask_cluster import create_task_runner
from morflowgenesis.utils.image_object import ImageObject
from morflowgenesis.utils.step_output import StepOutput


@task
def generate_object(
    existing_ids,
    row,
    working_dir,
    step_name,
    source_column,
    non_source_columns,
    metadata_column=None,
):
    source_img = AICSImage(row[source_column])
    # add metadata
    metadata = {"S": source_img.scenes, "T": source_img.dims.T - 1, "C": source_img.dims.C - 1}
    if metadata_column is not None:
        metadata.update(json.loads(row[metadata_column].replace('\'', '\"')))

    obj = ImageObject(working_dir, row[source_column], metadata)
    if obj.id in existing_ids:
        print(f"ID {obj.id} already exists. Skipping...")
        return

    for col in [source_column] + non_source_columns:
        step_output = StepOutput(
            working_dir, step_name, col, "image", image_id=obj.id, path=row[col]
        )
        obj.add_step_output(step_output)
    obj.save()


@flow(task_runner=create_task_runner(), log_prints=True)
def generate_objects(
    working_dir,
    step_name,
    csv_path,
    source_column,
    non_source_columns=[],
    metadata_column=None,
):
    image_objects = [
        ImageObject.parse_file(obj_path)
        for obj_path in (Path(working_dir) / "_ImageObjectStore").glob("*")
    ]

    """Generate a new image object for each row in the csv file."""
    df = pd.read_csv(csv_path)

    existing_ids = [im_obj.id for im_obj in image_objects]
    new_image_objects = []
    for row in df.itertuples():
        new_image_objects.append(
            generate_object.submit(
                existing_ids,
                row._asdict(),
                working_dir,
                step_name,
                source_column,
                non_source_columns,
                metadata_column,
            )
        )

    [im_obj.result() for im_obj in new_image_objects]
