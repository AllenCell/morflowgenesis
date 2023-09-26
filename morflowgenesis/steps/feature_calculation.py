import numbers

import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from aicsshparam import shparam, shtools
from prefect import flow, task
from skimage.measure import label

from morflowgenesis.utils import create_task_runner
from morflowgenesis.utils.image_object import ImageObject
from morflowgenesis.utils.step_output import StepOutput


def get_volume(img):
    return img.sum()


def get_height(img):
    z, _, _ = np.where(img)
    return z.max() - z.min()


def get_n_pieces(img):
    return len(np.unique(label(img))) - 1


def get_shcoeff(img, transform_params=None, lmax=16, return_transform=False):
    alignment_2d = True
    if transform_params is not None:
        img = shtools.apply_image_alignment_2d(img, transform_params[-1])
        img = img[0]
        alignment_2d = False
    (coeffs, _), (_, _, _, transform_params) = shparam.get_shcoeffs(
        image=img, lmax=lmax, alignment_2d=alignment_2d
    )
    if return_transform:
        return coeffs, transform_params
    return coeffs


FEATURE_EXTRACTION_FUNCTIONS = {
    "volume": get_volume,
    "height": get_height,
    "shcoeff": get_shcoeff,
    "n_pieces": get_n_pieces,
}


@task
def get_features(row, features, segmentation_columns):
    data = {"CellId": row["CellId"]}
    for col in segmentation_columns:
        path = row[col]
        img = AICSImage(path)
        channel_names = img.channel_names
        img = img.get_image_dask_data("CZYX", S=0, T=0).compute()
        for i, name in enumerate(channel_names):
            for feat in features:
                if feat not in FEATURE_EXTRACTION_FUNCTIONS:
                    print(
                        f"Feature {feat} not found. Options are {FEATURE_EXTRACTION_FUNCTIONS.keys()}"
                    )
                    continue
                feats = FEATURE_EXTRACTION_FUNCTIONS[feat](img[i])
                if isinstance(feats, numbers.Number):
                    data[f"{feat}_{col}_{name}"] = feats
                elif isinstance(feats, dict):
                    for k, v in feats.items():
                        data[f"{feat}_{col}_{name}_{k}"] = v
    return pd.DataFrame([data])


@task()
def get_matched_features(row, features, segmentation_columns, reference_channel):
    data = {"CellId": row["CellId"]}
    for col in segmentation_columns:
        img = AICSImage(row[col])
        channel_names = img.channel_names
        img = img.data.squeeze()

        reference_idx = channel_names.index(reference_channel)
        for feat in features:
            if feat not in FEATURE_EXTRACTION_FUNCTIONS:
                print(
                    f"Feature {feat} not found. Options are {FEATURE_EXTRACTION_FUNCTIONS.keys()}"
                )
                continue
            if feat == "shcoeff":
                feats_label, transform_params = FEATURE_EXTRACTION_FUNCTIONS[feat](
                    img[reference_idx], return_transform=True
                )
                for k, v in feats_label.items():
                    data[f"{feat}_{col}_{reference_channel}_{k}"] = v
                for i, ch in enumerate(channel_names):
                    if ch != reference_channel:
                        feats_pred = FEATURE_EXTRACTION_FUNCTIONS[feat](img[i], transform_params)
                        for k, v in feats_pred.items():
                            data[f"{feat}_{col}_{ch}_{k}"] = v
            else:
                for i, name in enumerate(channel_names):
                    feats_label = FEATURE_EXTRACTION_FUNCTIONS[feat](img[i])
                    data[f"{feat}_{col}_{name}"] = feats_label
    return pd.DataFrame([data])


@flow(task_runner=create_task_runner(), log_prints=True)
def calculate_features(
    image_object_path,
    step_name,
    output_name,
    input_step,
    features,
    segmentation_columns,
    reference_channel=None,
):
    image_object = ImageObject.parse_file(image_object_path)

    cell_df = image_object.load_step(input_step)
    results = []
    for row in cell_df.itertuples():
        row = row._asdict()
        if reference_channel is None:
            results.append(get_features.submit(row, features, segmentation_columns))
        else:
            results.append(
                get_matched_features.submit(row, features, segmentation_columns, reference_channel)
            )

    features_df = pd.concat([r.result() for r in results])

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
