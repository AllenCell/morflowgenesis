import re
from typing import List

import pandas as pd
from sklearn.decomposition import PCA

from morflowgenesis.utils import ImageObject, StepOutput


def run_pca(
    image_objects: List[ImageObject],
    output_name: str,
    feature_step: str,
    features_regex: str,
    apply_names: List[str],
    calculate_name: str = None,
    n_components: int = 10,
    reference_features_path: str = None,
    tags: List[str] = [],
):
    """
    Run PCA on features from calculate_name, and apply to features from apply_names
    Parameters
    ----------
    image_objects : List[ImageObject]
        List of ImageObjects to run PCA on
    output_name : str
        Name of output
    feature_step : str
        Step name of calculated features
    features_regex : str
        Regular expression to match feature columns
    calculate_name : str
        Name of features to calculate PCA on
    apply_names : List[str]
        List of names to apply PCA to
    n_components : int
        Number of principal components to use
    tags : List[str]
        [UNUSED] Tags corresponding to concurrency-limits for parallel processing
    """
    if calculate_name in apply_names:
        raise ValueError("Calculate_name cannot be in apply_names")
    features = pd.concat([obj.load_step(feature_step) for obj in image_objects])
    feature_columns = [c for c in features.columns if re.search(features_regex, c)]
    if calculate_name is None:
        assert (
            reference_features_path is not None
        ), "Reference features path must be provided if calculate_name is None"
        print("Using reference features for PCA")
        target_df = pd.read_csv(reference_features_path)
    else:
        target_df = features.xs(calculate_name, level="Name")
    pca = PCA(n_components=n_components)
    pca.fit(target_df[feature_columns])

    x = pd.DataFrame()
    if calculate_name is not None:
        x = pca.transform(target_df[feature_columns])
        x = pd.DataFrame(x)
        x.columns = [f"PC{i+1}" for i in range(n_components)]
        x["CellId"] = target_df.index.get_level_values("CellId")
        x["Name"] = calculate_name

    for name in apply_names:
        pred_df = features.xs(name, level="Name")
        y = pca.transform(pred_df[feature_columns])
        y = pd.DataFrame(y)
        y.columns = [f"PC{i+1}" for i in range(n_components)]
        y["CellId"] = pred_df.index.get_level_values("CellId")
        y["Name"] = name
        x = pd.concat([x, y])
    x = x.set_index(["CellId", "Name"])

    step_output = StepOutput(
        image_objects[0].working_dir,
        "run_pca",
        output_name,
        "csv",
        image_id="pca",
        index_col=["CellId", "Name"],
    )
    step_output.save(x)
    for obj in image_objects:
        obj.add_step_output(step_output)
        obj.save()
