from pathlib import Path
import json

from aicsimageio import AICSImage
from prefect import flow, task

from morflowgenesis.utils import create_task_runner
from morflowgenesis.utils.image_object import ImageObject, StepOutput

from camera_alignment_core import Align, Magnification
from camera_alignment_core.channel_info import channel_info_factory, CameraPosition
from camera_alignment_core.alignment_core import align_image, crop


def get_data(img, load_kwargs):
    # keep s when load kwargs is used later
    load_kwargs = load_kwargs.copy()
    if "S" in load_kwargs:
        img.set_scene(load_kwargs.pop("S"))
    data = img.get_image_dask_data(**load_kwargs).compute()
    return data


@task
def split(img, working_dir, output_name, step_name, file_path, alignment_args, load_kwargs):
    # load image
    data = get_data(img, load_kwargs)

    if alignment_args is not None:
        data = align_image(data, alignment_args['matrix'], channels_to_shift=alignment_args['channels'])
        data = crop(data, Magnification(20))

    # create object associated with image
    img_obj = ImageObject(working_dir, file_path, load_kwargs)
    output = StepOutput(working_dir, step_name, output_name, "image", image_id=img_obj.id)
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
    optical_control_back_channels = (
        optical_control_channel_info.channels_from_camera_position(CameraPosition.BACK)
    )

    align.align_optical_control(
        channels_to_shift=[
            channel.channel_index for channel in optical_control_back_channels
        ],
    )
    return align.alignment_transform.matrix

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
    dimension_order_out="CZYX",
    optical_control_path=None,
):
    working_dir = Path(working_dir)
    (working_dir / step_name).mkdir(exist_ok=True, parents=True)

    alignment_args ={}
    if optical_control_path is not None:
        alignment_args ={}
        alignment_args['matrix']= align_argolight(working_dir / 'optical_control_alignment', optical_control_path)
        alignment_channels = channel_info_factory(czi_path).channels_from_camera_position(
            CameraPosition.BACK
        )
        alignment_args['channels']=[channel.channel_index for channel in alignment_channels]
        with open(working_dir / 'optical_control_alignment'/'alignment_params.json', 'w') as f:
            json.dump(str(alignment_args), f)
        print('Alignment Parameters:', alignment_args)

    # get source image metadata
    img = AICSImage(czi_path)
    scenes = img.scenes if scenes == -1 else scenes

    # run all timepoints if no timepoints specified in config
    timepoints = range(img.dims.T) if timepoints == -1 else timepoints
    timepoints = _validate_list(timepoints)
    # same for channels
    channels = range(img.dims.C) if channels == -1 else channels
    channels = _validate_list(channels)

    image_objects = [ImageObject.parse_file(obj_path) for obj_path in (working_dir/ "_ImageObjectStore").glob('*.json')]
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
                    split.submit(img, working_dir, output_name, step_name, czi_path, alignment_args, load_kwargs)
                )
            else:
                print(f"Scene {s} timepoint {t} already run")
    [im_obj.result() for im_obj in new_image_objects]

if __name__ == '__main__':
    path= "//allen/aics/assay-dev/users/Benji/hydra_workflow/workings/_ImageObjectStore/04d52c5f39b53bf6f518ea7f73fe0bb209a1796529d6599bfe8cc30a.json"
    split_czi(
        czi_path = '/allen/aics/assay-dev/MicroscopyData/Frick/2023/20230327/ZSD0/AICS86_day3_nucleolus_movie_interactive-01-1.czi/AICS86_day3_nucleolus_movie_interactive-01-1_AcquisitionBlock1.czi/AICS86_day3_nucleolus_movie_interactive-01-1_AcquisitionBlock1_pt1.czi',
        working_dir= '/allen/aics/assay-dev/users/Benji/hydra_workflow/workings',
        output_name='20x_raw',
        step_name='split_czi',
        scenes=['P4-F6'],
        timepoints=[0,1],
        channels=-1,
        dimension_order_out="CZYX",
        optical_control_path="//allen/aics/assay-dev/MicroscopyData/Frick/2023/20230327/ZSD0/QC/field_of_rings/20230327_ZSD0_SLG-506_20x_field_of_rings.czi"
    )