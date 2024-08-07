from typing import List

import pandas as pd
from aics_shape_modes.projection import (
    compute_pca_on_reps,
    write_all_shape_modes_latent_walk,
)

from morflowgenesis.utils import ImageObject, StepOutput


def make_shape_space(
    image_objects: List[ImageObject],
    output_name: str,
    feature_step: str,
    segmentation_names: List[str],
    tags: List[str],
    n_pcs: int = 10,
):
    """
    Create shape space from spherical harmonic features
    Parameters
    ----------
    image_objects : List[ImageObject]
        List of ImageObjects to create shape space from
    output_name : str
        Name of output
    feature_step : str
        Step name of calculated features
    segmentation_names : List[str]
        List of segmentation names to use for creating shape space
    tags : List[str]
        Tags corresponding to concurrency-limits for parallel processing
    n_pcs : int
        Number of principal components to use for shape space
    """
    features = pd.concat([obj.load_step(feature_step) for obj in image_objects])

    for seg_name in segmentation_names:
        step_output = StepOutput(
            image_objects[0].working_dir,
            output_name,
            "make_shape_space",
            "image",
            image_id="shape_modes_{seg_name}",
        )

        features = features.xs(seg_name, level="Name")
        shcoeffs = features[[col for col in features.columns if "shcoeff" in col]].to_numpy()

        pca, _, sm_df = compute_pca_on_reps(
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
