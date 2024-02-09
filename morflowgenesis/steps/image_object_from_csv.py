import json
from pathlib import Path
from typing import List, str

import pandas as pd
from aicsimageio import AICSImage

from morflowgenesis.utils import ImageObject, StepOutput, parallelize_across_images


def generate_object(
    row,
    existing_ids,
    working_dir,
    source_column,
    non_source_columns,
    metadata_column=None,
):
    row = row._asdict()
    source_img = AICSImage(row[source_column])
    # add metadata
    metadata = {"S": source_img.scenes, "T": source_img.dims.T - 1, "C": source_img.dims.C - 1}
    if metadata_column is not None:
        metadata.update(json.loads(row[metadata_column].replace("'", '"')))

    obj = ImageObject(working_dir, row[source_column], metadata)
    if obj.id in existing_ids:
        print(f"ID {obj.id} already exists. Skipping...")
        return

    for col in [source_column] + non_source_columns:
        step_output = StepOutput(
            working_dir, "generate_objects", col, "image", image_id=obj.id, path=row[col]
        )
        obj.add_step_output(step_output)
    obj.save()


def generate_objects(
    working_dir: str,
    csv_path: str,
    source_column: str,
    non_source_columns: List[str] = [],
    metadata_column: str = None,
    image_objects: List[ImageObject] = [],
    tags: List[str] = [],
):
    """
    Generate image objects from a csv file
    Parameters
    ----------
    working_dir : str
        Working directory to save image objects
    csv_path : str
        Path to csv file
    source_column : str
        Column name of source image
    non_source_columns : List[str], optional
        List of column names to save as additional step outputs
    metadata_column : str, optional
        Column name of metadata. Metadata should be in json format
    image_objects : List[ImageObject], optional
        List of existing ImageObjects
    tags : List[str], optional
        Tags corresponding to concurrency-limits for parallel processing
    """
    existing_ids = [obj.id for obj in image_objects]

    """Generate a new image object for each row in the csv file."""
    df = pd.read_csv(csv_path)
    parallelize_across_images(
        df.itertuples(),
        generate_object,
        tags=tags,
        data_name="row",
        existing_ids=existing_ids,
        working_dir=working_dir,
        source_column=source_column,
        non_source_columns=non_source_columns,
        metadata_column=metadata_column,
    )
