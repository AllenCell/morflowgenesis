from pathlib import Path

import numpy as np
import pandas as pd
from cyto_dl.api import CytoDLModel
from skimage.transform import resize
from prefect import flow, task
from aicsimageio.writers import OmeTiffWriter
from morflowgenesis.utils import create_task_runner
from morflowgenesis.utils.image_object import ImageObject
from morflowgenesis.utils.step_output import StepOutput

@task()
def extract_fov_tracks(obj, image_step, single_cell_step, z_resize=2.5005, resize_shape=64):
    timepoint = obj.metadata['T']
    img = obj.load_step(image_step)
    img = (img - np.mean(img)) / np.std(img)

    single_cell_df = obj.load_step(single_cell_step)
    single_cell_df = single_cell_df[single_cell_df.index_sequence == timepoint]
    data = {}
    #  remove [], split on commas, z coords, resize to 20x coords, convert to int
    rois = (
        single_cell_df["roi"]
        .apply(lambda x: (np.array(x[1:-1].split(",")[2:], dtype=float) / z_resize).astype(int))
        .values
    )  #
    for i, row in enumerate(single_cell_df.itertuples()):
        crop = img[rois[i][0] : rois[i][1], rois[i][2] : rois[i][3]]
        data[row.track_id] = resize(
            crop, (resize_shape, resize_shape), anti_aliasing=True, preserve_range=True
        ).astype(np.float16)
    data["timepoint"] = timepoint
    print(f'Crops extracted from {timepoint}')
    return data

@task
def save_track_dataset(data, save_dir):
    # iterate by timepoint
    data_by_track = {}
    for timepoint_patch_dict in data:
        # patch metadata
        timepoint = timepoint_patch_dict["timepoint"]
        del timepoint_patch_dict["timepoint"]

        # patch data
        for track_id, patch in timepoint_patch_dict.items():
            if track_id not in data_by_track:
                data_by_track[track_id] = {
                    "img": [],
                    "track_start": int(timepoint),
                    "track_id": track_id,
                }
            data_by_track[track_id]["img"].append(patch)

    metadata = []
    # aggregate by track
    for track_id, data in data_by_track.items():
        img = np.stack(data["img"])

        metadata.append(
            {
                'track_start': data["track_start"],
                'track_id': track_id,
                'path': save_dir/f'{track_id}.tiff'
            }
        )
        OmeTiffWriter.save(uri = save_dir + f'{track_id}.tif', data = img.astype(float), dimension_order = 'CYX')
        print(f'{track_id} saved')
    metadata = pd.DataFrame(metadata)
    metadata.to_csv(save_dir + f"predict.csv")

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


@flow(task_runner=create_task_runner(), log_prints=True)
def formation_breakdown(image_object_paths, output_name, image_step, single_cell_step, config_path, overrides):
    image_objects = [ImageObject.parse_file(p) for p in image_object_paths]
    output_dir = Path(f"{image_objects[0].working_dir}/formation_breakdown/{output_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = []
    for obj in image_objects:
        data.append(extract_fov_tracks.submit(obj, image_step, single_cell_step))
    data = [d.result() for d in data]

    data_dir = output_dir / "data"
    save_track_dataset(data, data_dir)

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