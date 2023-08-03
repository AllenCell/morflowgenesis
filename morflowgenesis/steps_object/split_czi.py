from prefect import task, flow, get_run_logger
from morflowgenesis.utils.image_object import ImageObject, StepOutput
from aicsimageio import AICSImage
from pathlib import Path
logger = get_run_logger()


def get_data(img, load_kwargs):
    # keep s when load kwargs is used later
    load_kwargs=load_kwargs.copy()
    if 'S' in load_kwargs:
        img.set_scene(load_kwargs.pop('S'))
    data = img.get_image_dask_data(**load_kwargs).compute()
    return data

@task
def split(img, working_dir, output_name,step_name, file_path, load_kwargs):
    data = get_data(img, load_kwargs)

    img_obj = ImageObject(working_dir, file_path, load_kwargs)
    output = StepOutput(working_dir,step_name, output_name, 'image', image_id = img_obj.id)
    output.save(data)
    img_obj.add_step_output(output)
    return img_obj

def _validate_list(val):
    if isinstance(val, list):
        return val
    else:
        return list(val)

@flow
def split_czi(image_objects, czi_path, working_dir, output_name, step_name, scenes=-1, timepoints=-1, channels=-1, dimension_order_out = 'ZYX'):
    working_dir = Path(working_dir)
    (working_dir/step_name).mkdir(exist_ok=True, parents=True)
    img = AICSImage(czi_path)
    scenes = img.scenes if scenes==-1 else scenes
    timepoints = range(img.dims.T) if timepoints==-1 else timepoints
    timepoints = _validate_list(timepoints)
    channels = range(img.dims.C) if channels==-1 else channels
    channels = _validate_list(channels)

    already_run = [(im_obj.T, im_obj.S)  for im_obj in image_objects]

    new_image_objects = []
    for s in scenes:
        for t in timepoints:
            # for c in channels:
            load_kwargs = {'S':s, 'T':t, 'C':channels, 'dimension_order_out':dimension_order_out}
            if (t, s) not in already_run:
                new_image_objects.append(split.submit(img, working_dir, output_name, step_name, czi_path, load_kwargs))
            else:
                logger.info(f'Scene {s} timepoint {t} already run')
    new_image_objects =  [im_obj.result() for im_obj in new_image_objects]
    [im_obj.save() for im_obj in new_image_objects]
    return image_objects + new_image_objects







