import os
from pathlib import Path

from aicsimageio import AICSImage
from prefect import flow, task

from morflowgenesis.utils import create_task_runner
from morflowgenesis.utils.image_object import ImageObject, StepOutput


def get_data(img, load_kwargs):
    # keep s when load kwargs is used later
    load_kwargs = load_kwargs.copy()
    if "S" in load_kwargs:
        img.set_scene(load_kwargs.pop("S"))
    data = img.get_image_dask_data(**load_kwargs).compute()
    return data


@task
def split(img, working_dir, output_name, step_name, file_path, load_kwargs):
    # load image
    data = get_data(img, load_kwargs)
    # create object associated with image
    img_obj = ImageObject(working_dir, file_path, load_kwargs)
    output = StepOutput(working_dir, step_name, output_name, "image", image_id=img_obj.id)
    output.save(data)
    img_obj.add_step_output(output)
    img_obj.save()

def _validate_list(val):
    if isinstance(val, list):
        return val
    return list(val)


@flow(task_runner=create_task_runner(), log_prints=True)
def split_czi(
    czi_path,
    working_dir,
    output_name,
    step_name,
    scenes=-1,
    timepoints=-1,
    channels=-1,
    dimension_order_out="ZYX",
):
    working_dir = Path(working_dir)
    (working_dir / step_name).mkdir(exist_ok=True, parents=True)

    # get source image metadata
    img = AICSImage(czi_path)
    scenes = img.scenes if scenes == -1 else scenes

    # run all timepoints if no timepoints specified in config
    timepoints = range(img.dims.T) if timepoints == -1 else timepoints
    timepoints = _validate_list(timepoints)
    # same for channels
    channels = range(img.dims.C) if channels == -1 else channels
    channels = _validate_list(channels)

    image_objects = [ImageObject.parse_file(obj_path) for obj_path in working_dir.glob('*.json')]
    already_run = [
        (im_obj.metadata.get("T"), im_obj.metadata.get("S")) for im_obj in image_objects
    ]
    new_image_objects = []
    for s in scenes:
        for t in timepoints:
            load_kwargs = {
                "S": s,
                "T": t,
                "C": channels,
                "dimension_order_out": dimension_order_out,
            }
            if (t, s) not in already_run:
                new_image_objects.append(
                    split.submit(img, working_dir, output_name, step_name, czi_path, load_kwargs)
                )
            else:
                print(f"Scene {s} timepoint {t} already run")
    [im_obj.result() for im_obj in new_image_objects]


if __name__ == "__main__":
    split_czi()
