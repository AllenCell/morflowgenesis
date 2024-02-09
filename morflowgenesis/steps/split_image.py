import json
from pathlib import Path
from typing import List, Optional

from aicsimageio import AICSImage
from camera_alignment_core import Align, Magnification
from camera_alignment_core.alignment_core import align_image, crop
from camera_alignment_core.channel_info import CameraPosition, channel_info_factory
from prefect import task

from morflowgenesis.utils.image_object import ImageObject, StepOutput


def get_data(path, load_kwargs):
    img = AICSImage(path)
    # keep s when load kwargs is used later
    if "S" in load_kwargs:
        img.set_scene(load_kwargs["S"])
    data = img.get_image_dask_data(**{k: v for k, v in load_kwargs.items() if k != "S"}).compute()
    return data


@task
def split(path, working_dir, output_name, alignment_args, load_kwargs):
    # create object associated with image
    img_obj = ImageObject(working_dir, path, load_kwargs)
    output = StepOutput(working_dir, "split_image", output_name, "image", image_id=img_obj.id)

    # load image
    data = get_data(path, load_kwargs)

    if alignment_args is not None:
        data = align_image(
            data, alignment_args["matrix"], channels_to_shift=alignment_args["channels"]
        )
        data = crop(data, Magnification(20))

    output.save(data)
    img_obj.add_step_output(output)
    img_obj.save()


@task
def align_argolight(savedir, optical_control_path):
    align = Align(
        optical_control=optical_control_path,
        magnification=Magnification(20),
        out_dir=savedir,
    )
    optical_control_channel_info = channel_info_factory(optical_control_path)
    optical_control_back_channels = optical_control_channel_info.channels_from_camera_position(
        CameraPosition.BACK
    )

    align.align_optical_control(
        channels_to_shift=[channel.channel_index for channel in optical_control_back_channels],
    )
    return align.alignment_transform.matrix


def _validate_list(val):
    if isinstance(val, list):
        return val
    return list(val)


def split_image(
    image_path: str,
    working_dir: str,
    output_name: str,
    scenes: Optional[List[int]] = -1,
    timepoints: Optional[List[int]] = -1,
    channels: Optional[List[int]] = -1,
    dimension_order_out: str = "CZYX",
    optical_control_path: Optional[str] = None,
    image_objects: List[ImageObject] = None,
    tags: List[str] = [],
):
    """
    Generate ImageObjects from input image based on scenes, timepoints, and channels
    Parameters
    ----------
    image_path : str
        Path to input image
    working_dir : str
        Working directory to save outputs
    output_name : str
        Name of output
    scenes : List[int]
        List of scenes to run on, by default -1 to indicate all scenes
    timepoints : List[int]
        List of timepoints to run on, by default -1 to indicate all timepoints
    channels : List[int]
        List of channels to run on, by default -1 to indicate all channels
    dimension_order_out : str
        Dimension order of output image, by default "CZYX"
    optical_control_path : str, optional
        Path to optical control image, by default None. If passed, input image will be aligned to optical control
    image_objects : List[ImageObject]
        List of existing ImageObjects
    tags : List[str]
        [UNUSED]
    """
    working_dir = Path(working_dir)
    (working_dir / "split_image").mkdir(exist_ok=True, parents=True)

    alignment_args = None
    if optical_control_path is not None:
        alignment_args = {}
        alignment_args["matrix"] = align_argolight(
            working_dir / "optical_control_alignment" / output_name, optical_control_path
        )
        alignment_channels = channel_info_factory(image_path).channels_from_camera_position(
            CameraPosition.BACK
        )
        alignment_args["channels"] = [channel.channel_index for channel in alignment_channels]
        with open(
            working_dir / "optical_control_alignment" / output_name / "alignment_params.json", "w"
        ) as f:
            json.dump(str(alignment_args), f)
        print("Alignment Parameters:", alignment_args)

    # get source image metadata
    img = AICSImage(image_path)
    scenes = img.scenes if scenes == -1 else scenes

    # run all timepoints if no timepoints specified in config
    timepoints = range(img.dims.T) if timepoints == -1 else timepoints
    timepoints = _validate_list(timepoints)
    # same for channels
    channels = range(img.dims.C) if channels == -1 else channels
    channels = _validate_list(channels)

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
                    split.submit(
                        image_path,
                        working_dir,
                        output_name,
                        alignment_args,
                        load_kwargs.copy(),
                    )
                )
            else:
                print(f"Scene {s} timepoint {t} already run")
    [im_obj.result() for im_obj in new_image_objects]
