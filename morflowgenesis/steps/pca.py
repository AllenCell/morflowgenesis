import pandas as pd
from prefect import flow

from morflowgenesis.utils import create_task_runner
from morflowgenesis.utils.step_output import StepOutput
from morflowgenesis.utils.image_object import ImageObject
from sklearn.decomposition import PCA
import re
from typing import Dict, List



@flow(log_prints=True)
def run_pca(image_object_paths, step_name, output_name, feature_step, calculate_column: Dict[str, str], apply_columns: List[Dict[str, str]], n_components=10):
    image_objects = [ImageObject.parse_file(obj_path) for obj_path in image_object_paths]

    features = pd.concat([obj.load_step(feature_step) for obj in image_objects])
    pca = PCA(n_components=n_components)
    calculate_columns = [c for c in features.columns if re.search(calculate_column['regex'], c)]
    x = pca.fit_transform(features[calculate_columns].values)
    x = pd.DataFrame(x)
    x.columns= [f"PC{i+1}_{calculate_column['name']}" for i in range(n_components)]
    for apply_col in apply_columns:
        apply_columns = [c for c in features.columns if re.search( apply_col['regex'], c)]
        y = pca.transform(features[apply_columns].values)
        y = pd.DataFrame(y)
        y.columns= [f"PC{i+1}_{apply_col['name']}" for i in range(n_components)]
        x = pd.concat([x, y], axis=1)

    step_output = StepOutput(
        image_objects[0].working_dir, step_name, output_name, "csv", image_id='pca'
    )
    step_output.save(x)
    for obj in image_objects:
        obj.add_step_output(step_output)
        obj.save()
    
