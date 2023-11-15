import glob

import numpy as np
import pandas as pd
from aics_shape_modes.projection import (
    compute_pca_on_reps,
    write_all_shape_modes_latent_walk,
)
from prefect import flow

from morflowgenesis.utils import ImageObject, StepOutput, create_task_runner


@flow(log_prints=True)
def make_shape_space(
    image_object_paths, step_name, output_name, feature_step, segmentation_names, n_pcs=10
):
    image_objects = [ImageObject.parse_file(obj_path) for obj_path in image_object_paths]
    features = pd.concat([obj.load_step(feature_step) for obj in image_objects])

    for seg_name in segmentation_names:
        step_output = StepOutput(
            image_objects[0].working_dir,
            step_name,
            output_name,
            "image",
            image_id="shape_modes_{seg_name}",
        )

        features = features.xs(seg_name, level="Name")
        shcoeffs = features[[col for col in features.columns if "shcoeff" in col]].to_numpy()

        pca, axes, sm_df = compute_pca_on_reps(
            shcoeffs,
            projection_method="PCA",
            n_shapemodes=n_pcs,
            kernel_choice=None,
            kernel_gamma=None,
        )

        write_all_shape_modes_latent_walk(
            shape_modes_df=sm_df,
            pca=pca,
            out_dir=step_output.path.parent,
            comb_out_path=step_output.path,
            num_structures=1,
            sdfmesh_method=None,
            map_pts=[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
            use_she=True,
        )
