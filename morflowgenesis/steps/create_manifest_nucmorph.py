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
        cells = obj.load_step(single_cell_step)[
            [
                "CellId",
                "roi",
                "scale_micron",
                "centroid_z",
                "centroid_y",
                "centroid_x",
                "label_img",
                "seg_full_zstack_path",
            ]
        ]
        # HACK
        cells["raw_full_zstack_path"] = cells["seg_full_zstack_path"].apply(
            lambda x: x.replace("run_cytodl/nucseg", "split_image/split_image")
        )

        cells_with_feats = pd.merge(cells, features, on="CellId", how="outer")
        drop_cols = ["shcoeff"] + [
            col for col in cells_with_feats.columns if re.search("Unnamed", col)
        ]
        cells_with_feats = cells_with_feats.drop(columns=drop_cols, errors="ignore")
        cells_with_feats["index_sequence"] = obj.metadata["T"]
        manifest.append(cells_with_feats)
    manifest = pd.concat(manifest)

    # load tracking data
    tracking = obj.load_step(tracking_step)[
        ["index_sequence", "label_img", "fov_edge", "track_id", "lineage_id"]
    ]
    manifest = pd.merge(manifest, tracking, on=["index_sequence", "label_img"], how="left")

    if breakdown_classification_step in obj.steps:
        # add formation/breakdown information based on track_id
        # same for all objects, just load once
        breakdown_classification = obj.load_step(breakdown_classification_step)
        breakdown_classification = breakdown_classification[["track_id", "formation", "breakdown"]]
        manifest = pd.merge(manifest, breakdown_classification, on="track_id", how="left")

    # sort manifest
    manifest = manifest.sort_values(by="index_sequence")

    # rename columns
    manifest = manifest.rename(
        columns={col: f"NUC_{col}" for col in manifest.columns if re.search("shcoeff", col)}
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
