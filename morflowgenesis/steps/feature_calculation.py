import os

import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from aicsshparam import shparam, shtools
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect_dask import DaskTaskRunner

from morflowgenesis.utils.image_object import StepOutput

DASK_ADDRESS = os.environ.get("DASK_ADDRESS", None)


def get_volume(img):
    return img.sum()


def get_height(img):
    z, _, _ = np.where(img)
    return z.max() - z.min()


def get_shcoeff(img, transform_params=None, lmax=16):
    alignment_2d = True
    if transform_params is not None:
        img = shtools.apply_image_alignment_2d(img, transform_params[-1])
        img = img[0]
        alignment_2d = False
    (coeffs, _), (_, _, _, transform_params) = shparam.get_shcoeffs(
        image=img, lmax=lmax, alignment_2d=alignment_2d
    )
    return coeffs, transform_params


FEATURE_EXTRACTION_FUNCTIONS = {"volume": get_volume, "height": get_height, "shcoeff": get_shcoeff}


@task
def get_features(row, features, segmentation_columns):
    row = row._asdict()
    data = {"CellId": row["Cellid"]}
    for col in segmentation_columns:
        path = row[col]
        img = AICSImage(path).data.squeeze()
        for feat in features:
            if feat not in FEATURE_EXTRACTION_FUNCTIONS:
                print(
                    f"Feature {feat} not found. Options are {FEATURE_EXTRACTION_FUNCTIONS.keys()}"
                )
                continue
            feats = FEATURE_EXTRACTION_FUNCTIONS[feat](img)
            if isinstance(feats, (str, int, float)):
                data[f"{feat}_{col}"] = feats
            elif isinstance(feats, dict):
                for k, v in feats.items():
                    data[f"{feat}_{col}_{k}"] = v
    return pd.DataFrame(data)


@task
def get_matched_features(row_pred, row_label, features, segmentation_columns):
    data = {"CellId_pred": row_pred["CellId"], "CellId_label": row_label["CellId"]}
    for col in segmentation_columns:
        img_pred = AICSImage(row_pred[col]).data.squeeze()
        img_label = AICSImage(row_label[col]).data.squeeze()
        for feat in features:
            if feat not in FEATURE_EXTRACTION_FUNCTIONS:
                print(
                    f"Feature {feat} not found. Options are {FEATURE_EXTRACTION_FUNCTIONS.keys()}"
                )
                continue
            if feat == "shcoeff":
                feats_label, transform_params = FEATURE_EXTRACTION_FUNCTIONS[feat](img_label)
                feats_pred, _ = FEATURE_EXTRACTION_FUNCTIONS[feat](img_pred, transform_params)
                for k, v in feats_label.items():
                    data[f"{feat}_{col}_{k}_label"] = v
                for k, v in feats_pred.items():
                    data[f"{feat}_{col}_{k}_pred"] = v
            else:
                feats_label = FEATURE_EXTRACTION_FUNCTIONS[feat](img_label)
                feats_pred = FEATURE_EXTRACTION_FUNCTIONS[feat](img_pred)
                data[f"{feat}_{col}_label"] = feats_label
                data[f"{feat}_{col}_pred"] = feats_pred

    return pd.DataFrame([data])


@flow(task_runner=DaskTaskRunner(address=DASK_ADDRESS) if DASK_ADDRESS else SequentialTaskRunner())
def calculate_features(
    image_object,
    step_name,
    output_name,
    input_step,
    features,
    segmentation_columns,
    matching_step=None,
    reference_step=None,
):
    if image_object.step_is_run(f"{step_name}_{output_name}"):
        print(f"Skipping step {step_name}_{output_name} for image {image_object.id}")
        return image_object
    cell_df = image_object.load_step(input_step)
    if reference_step is not None:
        reference_df = image_object.load_step(reference_step)
        matching_df = image_object.load_step(matching_step)
    results = []
    for row in cell_df.itertuples():
        row = row._asdict()

        if reference_step is None:
            results.append(get_features.submit(row, features, segmentation_columns))
        else:
            label_cellid = matching_df[matching_df["pred_cellid"] == row["CellId"]]["label_cellid"]
            if len(label_cellid) == 0:
                # no matching cell found
                continue
            row_label = reference_df[reference_df["CellId"] == label_cellid.iloc[0]].iloc[0]
            results.append(
                get_matched_features(row, row_label, features, segmentation_columns)
            )  # .submit
    # features_df = pd.concat([r.result() for r in results])
    features_df = pd.concat(results)

    output = StepOutput(
        image_object.working_dir,
        step_name,
        output_name,
        output_type="csv",
        image_id=image_object.id,
    )
    output.save(features_df)
    image_object.add_step_output(output)
    image_object.save()

    return image_object
