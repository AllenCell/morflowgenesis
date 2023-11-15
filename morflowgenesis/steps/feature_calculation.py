import numpy as np
import pandas as pd
import tqdm
from aicsimageio import AICSImage
from aicsshparam import shparam, shtools
from prefect import flow, task
from skimage.measure import label

from morflowgenesis.utils import create_task_runner
from morflowgenesis.utils.image_object import ImageObject
from morflowgenesis.utils.step_output import StepOutput


def get_volume(img):
    return {"volume": np.sum(img)}


def get_height(img):
    z, _, _ = np.where(img)
    return {"height": z.max() - z.min()}


def get_n_pieces(img):
    return {"n_pieces": len(np.unique(label(img))) - 1}


def get_shcoeff(img, transform_params=None, lmax=16, return_transform=False):
    alignment_2d = True
    if transform_params is not None:
        img = shtools.apply_image_alignment_2d(img, transform_params[-1])[0]
        alignment_2d = False
    try:
        (coeffs, _), (_, _, _, transform_params) = shparam.get_shcoeffs(
            image=img, lmax=lmax, alignment_2d=alignment_2d
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


FEATURE_EXTRACTION_FUNCTIONS = {
    "volume": get_volume,
    "height": get_height,
    "shcoeff": get_shcoeff,
    "n_pieces": get_n_pieces,
}


def append_dict(features_dict, new_dict):
    for k, v in new_dict.items():
        try:
            features_dict[k].append(v)
        except KeyError:
            features_dict[k] = [v]
    return features_dict


@task
def get_features(row, features):
    features_dict = {}
    multi_index = []

    img = AICSImage(row["crop_seg_path"])
    channel_names = img.channel_names
    img = img.get_image_data("CZYX")

    for i, name in enumerate(channel_names):
        ch_img = img[i]
        if np.all(ch_img == 0):
            continue
        for feat in features:
            if feat not in FEATURE_EXTRACTION_FUNCTIONS:
                print(
                    f"Feature {feat} not found. Options are {FEATURE_EXTRACTION_FUNCTIONS.keys()}"
                )
                continue
            feats = FEATURE_EXTRACTION_FUNCTIONS[feat](ch_img)
            append_dict(features_dict, feats)
        multi_index.append((row["CellId"], name))
    return pd.DataFrame(
        features_dict,
        index=pd.MultiIndex.from_tuples(multi_index, names=["CellId", "Name"]),
    )


@task()
def get_matched_features(row, features, reference_channel):
    features_dict = {}
    img = AICSImage(row["crop_seg_path"])
    channel_names = img.channel_names
    img = img.get_image_data("CZYX")
    for ch in range(img.shape[0]):
        if np.all(img[ch] == 0):
            return None

    reference_idx = channel_names.index(reference_channel)
    # move reference channel to the front
    channel_names = [channel_names[reference_idx]] + [
        ch for ch in channel_names if ch != reference_channel
    ]
    for feat in features:
        if feat not in FEATURE_EXTRACTION_FUNCTIONS:
            print(f"Feature {feat} not found. Options are {FEATURE_EXTRACTION_FUNCTIONS.keys()}")
            continue
        if feat == "shcoeff":
            for i, ch in enumerate(channel_names):
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


@task(log_prints=True)
def get_features_gather(
    path, step_name, output_name, input_step, reference_channel, features
):
    image_object = ImageObject.parse_file(path)

    cell_df = image_object.load_step(input_step)
    results = []
    for row in tqdm.tqdm(cell_df.itertuples()):
        row = row._asdict()
        if reference_channel is None:
            results.append(get_features.fn(row, features))
        else:
            results.append(get_matched_features.fn(row, features, reference_channel))

    features_df = pd.concat(results)

    output = StepOutput(
        image_object.working_dir,
        step_name,
        output_name,
        output_type="csv",
        image_id=image_object.id,
        index_col=["CellId", "Name", "Label"],
    )
    output.save(features_df)
    image_object.add_step_output(output)
    image_object.save()


@flow(task_runner=create_task_runner(), log_prints=True)
def calculate_features(
    image_object_paths,
    step_name,
    output_name,
    input_step,
    features,
    reference_channel=None,
):
    results = []
    for p in image_object_paths:
        results.append(
            get_features_gather.submit(
                p, step_name, output_name, input_step, reference_channel, features
            )
        )
        break
    [r.result() for r in results]
