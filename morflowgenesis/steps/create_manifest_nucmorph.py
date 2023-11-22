import re
from typing import List

import pandas as pd
from prefect import flow

from morflowgenesis.utils.image_object import ImageObject
from morflowgenesis.utils.step_output import StepOutput


@flow(log_prints=True)
def create_manifest(
    image_object_paths,
    step_name,
    output_name,
    feature_step,
    tracking_step,
    single_cell_step,
):
    image_objects = [ImageObject.parse_file(obj_path) for obj_path in image_object_paths]

    tracking_df = image_objects[0].load_step(tracking_step)
    manifest = []
    for obj in image_objects:
        tracks = tracking_df[tracking_df.time_index == obj.metadata['T']]
        tracks =tracks[['centroid_z', 'centroid_y', 'centroid_x', 'label_img', 'time_index', 'is_outlier', 'has_outlier', 'past_outlier', 'normal_migration', 'edge_cell']]
        features = obj.load_step(feature_step)
        cells = obj.load_step(single_cell_step)
        assert len(features) == len(cells)
        assert len(cells) == len(tracks)
        tracked_cells = pd.merge(tracks, cells, on=["label_img",'centroid_z', 'centroid_y', 'centroid_x'])
        tracked_cells_with_feats = pd.merge(tracked_cells, features, on="CellId")        
        manifest.append(tracked_cells_with_feats)

    manifest = pd.concat(manifest)
    manifest = manifest.sort_values(by="time_index")
    step_output = StepOutput(
        image_objects[0].working_dir,
        step_name,
        output_name,
        "csv",
        image_id='manifest'
    )
    step_output.save(manifest)
    for obj in image_objects:
        obj.add_step_output(step_output)
        obj.save()

