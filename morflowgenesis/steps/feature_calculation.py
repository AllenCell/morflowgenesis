import numpy as np
import pandas as pd
import tqdm
from aicsimageio import AICSImage
from aicsshparam import shparam, shtools
from prefect import flow, task
from prefect.futures import PrefectFuture
from skimage.measure import label

from morflowgenesis.utils import ImageObject, StepOutput, create_task_runner, submit

# TODO either return everything in pixels or everything in microns
# TODO add centroid calculation


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


@task
def run_object(
    image_object,
    step_name,
    output_name,
    input_step,
    reference_channel,
    features,
    run_within_object,
):
    """General purpose function to run a task across an image object.

    If run_within_object is True, run the task across cells within the image object and return a
    list of futures and the output object. Otherwise, run the task as a function and return the
    results and an output object
    """
    cell_df = image_object.load_step(input_step)
    results = []
    for row in tqdm.tqdm(cell_df.itertuples()):
        row = row._asdict()
        if reference_channel is None:
            results.append(
                submit(get_features, as_task=run_within_object, row=row, features=features)
            )
        else:
            results.append(
                submit(
                    get_matched_features,
                    as_task=run_within_object,
                    row=row,
                    features=features,
                    reference_channel=reference_channel,
                )
            )

    output = StepOutput(
        image_object.working_dir,
        step_name,
        output_name,
        output_type="csv",
        image_id=image_object.id,
        index_col=["CellId", "Name", "Label"],
    )
    return results, output


@flow(task_runner=create_task_runner(), log_prints=True)
def calculate_features(
    image_object_paths,
    step_name,
    output_name,
    input_step,
    features,
    reference_channel=None,
):
    # if only one image is passed, run across objects within that image. Otherwise, run across images
    image_objects = [ImageObject.parse_file(path) for path in image_object_paths]
    run_within_object = len(image_objects) == 1

    all_results = []
    for obj in image_objects:
        all_results.append(
            submit(
                run_object,
                as_task=not run_within_object,
                image_object=obj,
                step_name=step_name,
                output_name=output_name,
                input_step=input_step,
                reference_channel=reference_channel,
                features=features,
                run_within_object=run_within_object,
            )
        )
    for object_result, obj in zip(all_results, image_objects):
        if not run_within_object:
            # parallelizing across fovs
            object_result = object_result.result()
        results, output = object_result
        if run_within_object:
            # parallelizing within fov
            results = [r.result() for r in results]
        features_df = pd.concat(results)
        output.save(features_df)
        obj.add_step_output(output)
        obj.save()
