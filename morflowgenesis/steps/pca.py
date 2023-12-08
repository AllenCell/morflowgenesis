import re
from typing import List

import pandas as pd
from prefect import flow
from sklearn.decomposition import PCA

from morflowgenesis.utils import ImageObject, StepOutput


@flow(log_prints=True)
def run_pca(
    image_object_paths,
    output_name,
    feature_step,
    features_regex,
    calculate_name: str,
    apply_names: List,
    n_components=10,
):
    image_objects = [ImageObject.parse_file(obj_path) for obj_path in image_object_paths]

    features = pd.concat([obj.load_step(feature_step) for obj in image_objects])
    feature_columns = [c for c in features.columns if re.search(features_regex, c)]
    target_df = features.xs(calculate_name, level="Name")

    pca = PCA(n_components=n_components)
    x = pca.fit_transform(target_df[feature_columns])
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
