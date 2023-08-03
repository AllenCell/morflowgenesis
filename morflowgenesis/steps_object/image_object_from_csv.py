from morflowgenesis.utils.image_object import ImageObject
from prefect import task, get_run_logger, flow
from prefect.task_runners import ConcurrentTaskRunner
import pandas as pd

logger = get_run_logger()


@task
def generate_object(row, working_dir, source_column, target_column, metadata_column=None):
    obj = ImageObject(working_dir, row[source_column], row.get(metadata_column))
    obj.target_column= row[target_column]
    return obj


@flow(task_runner=ConcurrentTaskRunner())
def generate_objects(image_objects, working_dir, csv_path, source_column, target_column,metadata_column=None):
    """
    Generate a new image object for each row in the csv file. 
    """
    df = pd.read_csv(csv_path)

    already_run = [im_obj.source_path for im_obj in image_objects]
    new_image_objects = []
    for row in df.itertuples():
        row = row._asdict()
        if row[source_column] not in already_run:
            new_image_objects.append(generate_object.submit(row, working_dir, source_column, target_column, metadata_column))
    new_image_objects =  [im_obj.result() for im_obj in new_image_objects]
    return image_objects + new_image_objects



