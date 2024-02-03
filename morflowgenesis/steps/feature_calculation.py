from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import tqdm
import vtk
from aicsimageio import AICSImage
from aicsshparam import shparam, shtools
from skimage.measure import label
from torchmetrics.classification import BinaryF1Score

from morflowgenesis.utils import ImageObject, StepOutput, parallelize_across_images


def get_surface_area(img, sigma=2):
    mesh, _, _ = shtools.get_mesh_from_image(image=img, sigma=sigma)

    massp = vtk.vtkMassProperties()
    massp.SetInputData(mesh)
    massp.Update()
    return {"mesh_vol": massp.GetVolume(), "mesh_sa": massp.GetSurfaceArea()}


def get_centroid(img):
    z, y, x = np.where(img)
    return {"centroid": (z.mean(), y.mean(), x.mean())}


def get_axis_lengths(img):
    # alignment adds a channel dimension
    img, _ = shtools.align_image_2d(img.copy(), compute_aligned_image=True)
    _, _, y, x = np.where(img)
    return {"width": np.ptp(y) + 1, "length": np.ptp(x) + 1}


def get_height_percentile(img):
    z, _, _ = np.where(img)
    return {"height_percentile": np.percentile(z, 99.9) - np.percentile(z, 0.1)}


def get_volume(img):
    return {"volume": np.sum(img)}


def get_height(img):
    z, _, _ = np.where(img)
    return {"height": z.max() - z.min()}


def get_n_pieces(img):
    return {"n_pieces": len(np.unique(label(img))) - 1}


def get_shcoeff(img, transform_params=None, lmax=16, sigma=2, return_transform=False):
    alignment_2d = True
    if transform_params is not None:
        img = shtools.apply_image_alignment_2d(img, transform_params[-1])[0]
        alignment_2d = False
    try:
        (coeffs, _), (_, _, _, transform_params) = shparam.get_shcoeffs(
            image=img, lmax=lmax, alignment_2d=alignment_2d, sigma=sigma
        )
    except ValueError as e:
        print(e)
        return {"shcoeff": None}

    if return_transform:
        return coeffs, transform_params
    return coeffs


def get_largest_cc(im):
    im = label(im)
    largest_cc = np.argmax(np.bincount(im.flatten())[1:]) + 1
    return im == largest_cc


def get_f1(img, reference):
    return {
        "f1": BinaryF1Score()(torch.from_numpy(img > 0), torch.from_numpy(reference > 0)).item()
    }


FEATURE_EXTRACTION_FUNCTIONS = {
    "volume": get_volume,
    "height": get_height,
    "height_percentile": get_height_percentile,
    "shcoeff": get_shcoeff,
    "n_pieces": get_n_pieces,
    "f1": get_f1,
    "surface_area": get_surface_area,
    "length_width": get_axis_lengths,
}


def compute_nucleolus_feats(img):
    nucleolus = img[0]

    nucleus = get_largest_cc(img[1])
    nucleus_vol = np.sum(nucleus)
    nucleus = np.max(nucleus, 0)
    nucleus = np.stack([nucleus] * nucleolus.shape[0], 0)

    nucleolus_volume = np.sum(np.logical_and(nucleolus, nucleus))
    return {
        "nucleus": {
            "volume": nucleus_vol,
        },
        "nucleolus": {
            "volume": nucleolus_volume,
        },
    }


def append_dict(features_dict, new_dict):
    for k, v in new_dict.items():
        try:
            features_dict[k].append(v)
        except KeyError:
            features_dict[k] = [v]
    return features_dict


def ensure_channel_first(img):
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    return img


def get_valid_features(features):
    # remove invalid features and warn user
    valid_features = [feat for feat in features if feat in FEATURE_EXTRACTION_FUNCTIONS]
    invalid_features = set(features) - set(valid_features)
    if len(invalid_features) > 0:
        print(
            f"Features {invalid_features} not found. Options are {FEATURE_EXTRACTION_FUNCTIONS.keys()}"
        )
    return valid_features


def get_channels(img, channel_names, run_channels):
    """
    Parameters
    ----------
    img: np.ndarray
        Image data in CZYX format
    channel_names: List[str]
        List of channel names in img
    run_channels: List[str]
        List of channels to run feature extraction on. If None, use all channels
    """
    run_channels = run_channels or channel_names
    if np.all([isinstance(ch, str) for ch in run_channels]):
        # create tuple of (index, channel_name) for non-zero channels to run
        channel_names = [
            (channel_names.index(c), c)
            for c in run_channels
            if c in channel_names and not np.all(img[channel_names.index(c)] == 0)
        ]
    elif np.all([isinstance(ch, int) for ch in run_channels]):
        channel_names = [(ch, f"Ch:{ch}") for ch in run_channels if not np.all(img[ch] == 0)]
    return channel_names


def get_roi_features(img, features, channels):
    features = get_valid_features(features)
    img = ensure_channel_first(img)
    channels = channels or range(img.shape[0])

    valid_channels = get_channels(img, channel_names=None, run_channels=channels)

    if len(valid_channels) == 0:
        print("No valid channels found!")
        return None

    features_dict = {}
    for i, name in valid_channels:
        for feat in features:
            append_dict(features_dict, FEATURE_EXTRACTION_FUNCTIONS[feat](img[i]))
    return pd.DataFrame(features_dict)


def get_matched_roi_features(img, features, channels, reference):
    features = get_valid_features(features)

    img = ensure_channel_first(img)
    reference = ensure_channel_first(reference)
    channels = channels or range(img.shape[0])
    valid_channels = get_channels(img, channel_names=None, run_channels=channels)

    if len(valid_channels) == 0:
        print("No valid channels found!")
        return None

    features_dict = {}
    for i, name in valid_channels:
        for feat in features:
            append_dict(features_dict, FEATURE_EXTRACTION_FUNCTIONS[feat](img[i], reference[i]))
    return pd.DataFrame(features_dict)


def get_cell_features(row, features, channels):
    row = row._asdict()
    valid_features = get_valid_features(features)

    img = AICSImage(row["crop_seg_path"])
    channel_names = img.channel_names
    img = img.get_image_data("CZYX")

    valid_channels = get_channels(img, channel_names, channels)

    if len(valid_channels) == 0:
        print("No valid channels found!")
        return None

    features_dict = {}
    multi_index = [(row["CellId"], name) for i, name in valid_channels]

    for i, name in valid_channels:
        for feat in valid_features:
            append_dict(features_dict, FEATURE_EXTRACTION_FUNCTIONS[feat](img[i]))

    try:
        features = pd.DataFrame(
            features_dict,
            index=pd.MultiIndex.from_tuples(multi_index, names=["CellId", "Name"]),
        )
        print("Cell", row["CellId"], "complete")
        return features
    except:
        return None


def get_matched_cell_features(row, features, channels, reference_channel):
    row = row._asdict()
    valid_features = get_valid_features(features)

    features_dict = {}
    img = AICSImage(row["crop_seg_path"])
    channel_names = img.channel_names
    channels = channels or channel_names
    img = img.get_image_data("CZYX")
    for ch in range(img.shape[0]):
        if np.all(img[ch] == 0):
            return None

    reference_idx = channel_names.index(reference_channel)
    # move reference channel to the front
    channel_names = [channel_names[reference_idx]] + [
        ch for ch in channel_names if ch != reference_channel
    ]
    for feat in valid_features:
        if feat == "shcoeff":
            for i, ch in enumerate(channel_names):
                if ch not in channels:
                    continue
                if i == 0:
                    # add ground truth features
                    feats, transform_params = FEATURE_EXTRACTION_FUNCTIONS[feat](
                        img[i], return_transform=True
                    )
                else:
                    feats, _ = FEATURE_EXTRACTION_FUNCTIONS[feat](img[i], transform_params)
                append_dict(features_dict, feats)
        else:
            for i, name in enumerate(channel_names):
                feats = FEATURE_EXTRACTION_FUNCTIONS[feat](img[i])
                append_dict(features_dict, feats)

    multi_index = [(row["CellId"], ch) for ch in channel_names]

    return pd.DataFrame(
        features_dict,
        index=pd.MultiIndex.from_tuples(multi_index, names=["CellId", "Name"]),
    )


def get_objects(object, input_step, reference_step):
    input = object.load_step(input_step)
    if isinstance(input, pd.DataFrame):
        return input.itertuples(), None, pd.DataFrame

    reference_step = object.load_step(reference_step) if reference_step is not None else None
    return [input], reference_step, np.array


def create_output(image_object, output_name, results):
    image_df = pd.concat(results)
    step_output = StepOutput(
        image_object.working_dir,
        "calculate_features",
        output_name,
        output_type="csv",
        image_id=image_object.id,
    )
    step_output.save(image_df)
    return step_output


def process_object(
    image_object, input_step, output_name, features, reference_channel, channels, reference_step
):
    data, reference, data_type = get_objects(image_object, input_step, reference_step)
    if data_type == pd.DataFrame:
        if reference_channel is not None:
            features = [
                get_matched_cell_features(d, features, channels, reference_channel) for d in data
            ]
        else:
            features = [get_cell_features(d, features, channels) for d in data]
    if data_type == np.array:
        if reference is not None:
            features = [get_matched_roi_features(d, features, channels, reference) for d in data]
        else:
            features = [get_roi_features(d, features) for d in data]
    return create_output(image_object, output_name, features)


def calculate_features(
    image_object_paths: List[Union[str, Path]],
    tags: List[str],
    run_type: str,
    output_name: str,
    input_step: str,
    features: List[str],
    reference_channel: Optional[str] = None,
    channels: Optional[List[str]] = None,
    reference_step: Optional[str] = None,
):
    """
    Parameters
    ----------
    image_object_paths: List[Union(str, Path)]
        List of paths to image objects to run the task on
    output_name: str
        Name of the output
    input_step: str
        Name of the input step (a single cell dataset or segmentation step)
    features: List[str]
        List of names of features to calculate
    reference_channel: Optional[str]
        Name of the reference channel to use for spherical harmonics alignment
    channels: Optional[List[str]]
        List of channel names to use for feature calculation. If None, use all channels
    reference_step: Optional[str]
        For FOV-based features, another image can be used to calculate FOV similarity features (like F1, Dice, etc.)
    """
    # if only one image is passed, run across objects within that image. Otherwise, run across images
    image_objects = [
        ImageObject.parse_file(path)
        for path in tqdm.tqdm(image_object_paths, desc="Loading image objects")
    ]

    if run_type == "images":
        parallelize_across_images(
            image_objects,
            process_object,
            tags=tags,
            input_step=input_step,
            output_name=output_name,
            features=features,
            reference_channel=reference_channel,
            channels=channels,
            reference_step=reference_step,
        )
