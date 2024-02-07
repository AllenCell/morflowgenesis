from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from copy import deepcopy
from aicsimageio.writers import OmeTiffWriter
from cyto_dl.api import CytoDLModel
from prefect import task
from skimage.transform import resize

from morflowgenesis.utils import ImageObject, StepOutput, parallelize_across_images


def extract_fov_tracks(
    data,
    image_step,
    resize_shape=64,
):
    image_object, rois = data
    timepoint = image_object.metadata["T"]
    img = image_object.load_step(image_step).max(0)
    img = (img - np.mean(img)) / np.std(img)
    rois = rois[rois.index_sequence == timepoint]

    data = {track_id: 
        resize(
            img[roi[0] : roi[1], roi[2] : roi[3]], 
            (resize_shape, resize_shape), 
            anti_aliasing=True, 
            preserve_range=True
        ).astype(np.float16) for roi, track_id in zip(rois['roi'], rois['track_id'])
    }
    data["timepoint"] = timepoint
    print(f"Crops extracted from {timepoint}")
    return data

def pad(df, t_max, pad =3):
    t0 = df.index_sequence.min()
    row_min = df[df.index_sequence == t0].iloc[0].to_dict()

    t1 = df.index_sequence.max()
    row_max = df[df.index_sequence == t1].iloc[0].to_dict()

    new_df = []
    for i in range(-pad, 0, 1):
        pad_timepoint= t0 + i
        if pad_timepoint >= 0:
            temp = deepcopy(row_min)
            temp['index_sequence']= pad_timepoint
            new_df.append(temp)

    for i in range(1, pad+1, 1):
        pad_timepoint= t1 + i
        if pad_timepoint < t_max:
            temp = deepcopy(row_max)
            temp['index_sequence'] =  pad_timepoint
            new_df.append(temp)
    new_df = pd.concat([df, pd.DataFrame(new_df)])
    return new_df

def get_rois(image_objects, single_cell_step, padding= 2, xy_resize=2.5005):
    """
        returns padded rois to extract from each timestep
    """
    print('Extracting ROIs')
    single_cell_df = pd.concat([obj.load_step(single_cell_step)[['roi', 'track_id', 'index_sequence']] for obj in image_objects])
    #  remove [], split on commas, z coords, resize to 20x coords, convert to int
    single_cell_df['roi'] = (
        single_cell_df["roi"]
        .apply(lambda x: (np.array(x[1:-1].split(",")[2:], dtype=float) / xy_resize).astype(int))
        .values
    )
    t_max = single_cell_df.index_sequence.max()
    print('Padding ROIs')
    single_cell_df = single_cell_df.groupby('track_id').apply(lambda x: pad(x, t_max, pad = padding))
    return single_cell_df

def save_track(data, save_dir):
    track_id, data = data
    if len(data["img"]) < 50:
        return
    metadata = {
        "track_start": data["track_start"],
        "timepoints": data["timepoints"],
        "track_id": track_id,
        "img": save_dir / f"{track_id}.tif",
    }
    OmeTiffWriter.save(
        uri=metadata["img"], data=np.stack(data["img"]).astype(float), dimension_order="CYX"
    )
    print(f"{track_id} saved")
    return pd.DataFrame([metadata])


def aggregate_by_track(data):
    # iterate by timepoint
    tp_order = np.argsort([d["timepoint"] for d in data])
    data_by_track = {}
    for idx in tqdm.tqdm(tp_order):
        timepoint_patch_dict = data[idx]
        timepoint = timepoint_patch_dict["timepoint"]
        # patch metadata
        del timepoint_patch_dict["timepoint"]

        # patch data
        for track_id, patch in timepoint_patch_dict.items():
            if track_id not in data_by_track:
                data_by_track[track_id] = {
                    "img": [],
                    "track_start": int(timepoint),
                    "track_id": track_id,
                    "timepoints": [],
                }
            data_by_track[track_id]["img"].append(patch)
            data_by_track[track_id]["timepoints"].append(timepoint)
    return data_by_track


@task
def load_model(
    save_dir,
    config_path,
    overrides,
):
    model = CytoDLModel()
    model.load_config_from_file(config_path)

    overrides.update(
        {
            "data.path": save_dir,
            "model.save_dir": str(save_dir),
            "paths.output_dir": str(save_dir),
            "paths.work_dir": str(save_dir),
        }
    )
    model.override_config(overrides)
    return model


@task(retries=3, retry_delay_seconds=[10, 60, 120])
def run_evaluate(model):
    return model.predict()

def formation_breakdown(
    image_objects,
    tags,
    output_name,
    image_step,
    single_cell_step,
    config_path,
    overrides,
):
    output_dir = Path(f"{image_objects[0].working_dir}/formation_breakdown/{output_name}")
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if not (data_dir / "predict.csv").exists():
        rois = get_rois(image_objects, single_cell_step)
        input_data = [(obj, rois[rois.index_sequence == obj.metadata['T']]) for obj in image_objects]
        _, data = parallelize_across_images(
            input_data,
            extract_fov_tracks,
            tags,
            create_output=False,
            image_step=image_step,
            data_name = "data",
        )
        data = aggregate_by_track(data)
        metadata = pd.concat([save_track(d, data_dir) for d in data.items()])
        metadata.to_csv(data_dir / "predict.csv")

    model = load_model(data_dir, config_path, overrides)
    _, _, out = run_evaluate(model)
    out = pd.DataFrame(out)
    output = StepOutput(
        image_objects[0].working_dir,
        step_name="formation_breakdown",
        output_name=output_name,
        output_type="csv",
        image_id="formation_breakdown_predictions",
    )
    output.save(out)

    for obj in image_objects:
        obj.add_step_output(output)
        obj.save()