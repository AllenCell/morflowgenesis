import re
from typing import List, Optional

import pandas as pd
import tqdm

from morflowgenesis.utils import ImageObject, StepOutput


def create_manifest(
    image_objects: List[ImageObject],
    output_name: str,
    dataset_name: str,
    feature_step: str,
    single_cell_step: str,
    tracking_step: str,
    breakdown_classification_step: Optional[str] = None,
    tags: List[str] = [],
):
    """Postprocessing function for combining results from nucmorph pipeline into single
    manifest."""
    # load features and single cell data
    manifest = []
    for obj in tqdm.tqdm(image_objects, desc="Loading single cell features..."):
        features = obj.load_step(feature_step)
        cells = obj.load_step(single_cell_step)
        cells_with_feats = pd.merge(cells, features, on="CellId", how="outer")
        drop_cols = ["shcoeff", "name_dict", "channel_names_seg"] + [
            col for col in cells_with_feats.columns if re.search("Unnamed", col)
        ]
        cells_with_feats = cells_with_feats.drop(columns=drop_cols, errors="ignore")
        manifest.append(cells_with_feats)
    manifest = pd.concat(manifest)

    if breakdown_classification_step is not None:
        # add formation/breakdown information based on track_id
        # same for all objects, just load once
        breakdown_classification = obj.load_step(breakdown_classification_step)
        breakdown_classification = breakdown_classification[['track_id', 'formation', 'breakdown']]
        manifest = pd.merge(manifest, breakdown_classification, on="track_id", how="left")

    # load tracking data
    tracking = obj.load_step(tracking_step)[['index_sequence', 'label_img', 'edge_cell', 'track_id', 'lineage_id']]
    manifest = pd.merge(manifest, tracking, on=['index_sequence', 'label_img'], how="left")
    manifest = manifest.drop(columns=['index_sequence_y', 'label_img_y'])
    manifest = manifest.rename(columns={'index_sequence_x': 'index_sequence', 'label_img_x': 'label_img', 'edge_cell': 'fov_edge'})

    # sort manifest
    manifest = manifest.sort_values(by="index_sequence")

    # rename columns
    shcoeff_cols = [col for col in manifest.columns if re.search("shcoeff", col)]
    shcoeff_rename_dict = {col: f"NUC_{col}" for col in shcoeff_cols}
    manifest = manifest.rename(columns=shcoeff_rename_dict)
    manifest = manifest.rename(
        columns={
            "20x_lamin_path": "raw_full_zstack_path",
            "nuc_seg_path": "seg_full_zstack_path",
        }
    )
    manifest["dataset"] = dataset_name

    # save
    step_output = StepOutput(
        image_objects[0].working_dir, "create_manifest", output_name, "csv", image_id="manifest"
    )
    step_output.save(manifest)
    for obj in image_objects:
        obj.add_step_output(step_output)
        obj.save()
