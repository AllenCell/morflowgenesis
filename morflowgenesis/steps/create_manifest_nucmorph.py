import re
from typing import List

import pandas as pd
from prefect import flow

from morflowgenesis.utils.image_object import ImageObject
from morflowgenesis.utils.step_output import StepOutput


@flow(log_prints=True)
def create_manifest(
    image_object_paths,
    output_name,
    feature_step,
    tracking_step,
    single_cell_step,
):
    image_objects = [ImageObject.parse_file(obj_path) for obj_path in image_object_paths]

    tracking_df = image_objects[0].load_step(tracking_step)
    manifest = []
    for obj in image_objects:
        tracks = tracking_df[tracking_df.time_index == obj.metadata["T"]]
        tracks = tracks[
            [
                "centroid_z",
                "centroid_y",
                "centroid_x",
                "label_img",
                "time_index",
                "is_outlier",
                "has_outlier",
                "past_outlier",
                "normal_migration",
                "track_id",
            ]
        ]
        features = obj.load_step(feature_step)
        cells = obj.load_step(single_cell_step)
        assert len(features) == len(cells)
        assert len(cells) == len(tracks)
        tracked_cells = pd.merge(
            tracks, cells, on=["label_img", "centroid_z", "centroid_y", "centroid_x"]
        )
        tracked_cells_with_feats = pd.merge(tracked_cells, features, on="CellId")
        drop_cols = [col for col in tracked_cells_with_feats.columns if re.search("Unnamed", col)]
        tracked_cells_with_feats = tracked_cells_with_feats.drop(columns=drop_cols)
        manifest.append(tracked_cells_with_feats)

    manifest = pd.concat(manifest)
    manifest = manifest.sort_values(by="time_index")
    manifest = manifest.rename(
        columns={
            "time_index": "T_index",
            "20x_lamin_path": "raw_full_zstack_path",
            "nuc_seg_path": "seg_full_zstack_path",
        }
    )
    step_output = StepOutput(
        image_objects[0].working_dir, 'create_manifest', output_name, "csv", image_id="manifest"
    )
    step_output.save(manifest)
    for obj in image_objects:
        obj.add_step_output(step_output)
        obj.save()
