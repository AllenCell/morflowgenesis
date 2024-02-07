import re
from pathlib import Path
from typing import List, Union

import pandas as pd

from morflowgenesis.utils import ImageObject, StepOutput


def create_manifest(
    image_objects,
    output_name: str,
    feature_step: str,
    single_cell_step: str,
    dataset_name: str,
    # breakdown_classification_step: str,
    tags: List[str],
):
    # load features and single cell data
    manifest = []
    for obj in image_objects:
        features = obj.load_step(feature_step)
        cells = obj.load_step(single_cell_step)
        assert len(features) == len(cells)
        cells_with_feats = pd.merge(cells, features, on="CellId")
        drop_cols = ["shcoeff", "name_dict", "channel_names_seg"] + [
            col for col in cells_with_feats.columns if re.search("Unnamed", col)
        ]
        cells_with_feats = cells_with_feats.drop(columns=drop_cols, errors="ignore")
        manifest.append(cells_with_feats)
    manifest = pd.concat(manifest)

    # add formation/breakdown information based on track_id
    # same for all objects, just load once
    # breakdown_classification = obj.load_step(breakdown_classification_step)
    # manifest = pd.merge(manifest, breakdown_classification, on="track_id")

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
    manifest['dataset'] = dataset_name

    # save
    step_output = StepOutput(
        image_objects[0].working_dir, "create_manifest", output_name, "csv", image_id="manifest"
    )
    step_output.save(manifest)
    for obj in image_objects:
        obj.add_step_output(step_output)
        obj.save()