import numpy as np
from typing import Union, List
from prefect import flow, task
from scipy.optimize import curve_fit
from morflowgenesis.utils import ImageObject, StepOutput, create_task_runner

def gaussian(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

@task
def crop(
    image_object: ImageObject,
    output_name: str,
    image_step: str,
    pad: Union[int, List[int]] = 5,
    min_slices:int =24,
    sigma_cutoff: Union[int, List[int]] = 2,
):
    """
    Crop to center of an image assuming a Gaussian-like profile along the z-axis.
    """
    img = image_object.load_step(image_step)
    padding = np.array([(0,0), (0,0), (0,0)])
    if len(img.shape) != 3:
        raise ValueError("Image must be 3D")

    if img.shape[0] < min_slices:
        raise ValueError(f"Image must have at least {min_slices}, got {img.shape[0]}")
    elif img.shape[0] > min_slices:
        # only crop if image != min_slices
        # minmax normalize z-profile of standard deviation and fit gaussian
        z_prof = np.std(img, axis=(1, 2))
        z_prof = (z_prof - np.min(z_prof)) / (np.max(z_prof) - np.min(z_prof))
        popt, _ = curve_fit(gaussian, np.arange(len(z_prof)), z_prof, p0=[1, len(z_prof)/2, 0.3])
        _, center, sigma = popt

        if isinstance(sigma_cutoff, list):
            sigma_cutoff_low, sigma_cutoff_high = sigma_cutoff
        else:
            sigma_cutoff_low = sigma_cutoff_high = sigma_cutoff

        if isinstance(pad, list):
            pad_low, pad_high = pad
        else:
            pad_low = pad_high = pad

        # ensure crop is within bounds of image
        bottom_z = max(0, int(center - sigma * sigma_cutoff_low)-pad_low)
        top_z = min(img.shape[0], int(center + sigma * sigma_cutoff_high)+pad_high)

        num_slices = top_z - bottom_z

        # ensure crop is at least min_slices
        if num_slices < min_slices:
            if bottom_z == 0:
                top_z = min(img.shape[0], top_z + min_slices - num_slices)
            elif top_z == img.shape[0]:
                bottom_z = max(0, bottom_z - min_slices + num_slices)
            else:
                bottom_z = max(0, bottom_z - (min_slices - num_slices) // 2)
                top_z = min(img.shape[0], top_z + (min_slices - num_slices) // 2)
        padding[0] = [bottom_z,img.shape[0] - top_z]
        img = img[bottom_z : top_z]

    # add result to image object
    output = StepOutput(
        image_object.working_dir,
        step_name="center_crop",
        output_name=output_name,
        output_type="image",
        image_id=image_object.id,
    )
    output.save(img)
    np.save(image_object.working_dir / "center_crop"/output_name/ f"{image_object.id}.npy", padding)
    return output


@flow(task_runner=create_task_runner(), log_prints=True)
def center_crop(
    image_object_paths, image_step, output_name, pad = 5, min_slices = 24, sigma_cutoff = 2
):
    image_objects = [ImageObject.parse_file(path) for path in image_object_paths]

    results = []
    for obj in image_objects:
        results.append(
            crop.submit(
                obj,
                output_name,
                image_step,
                pad = pad,
                min_slices = min_slices,
                sigma_cutoff = sigma_cutoff,
            )
        )
    step_outputs = [result.result() for result in results]
    for obj, step_output in zip(image_objects, step_outputs):
        obj.add_step_output(step_output)
        obj.save()

@task(log_prints=True, tags = ['benji_50'])
def uncrop(image_object, output_name, image_step, cropping_step, mode = 'constant', pad_rescale = 1):
    img = image_object.load_step(image_step)
    print('image loaded')
    # crop path is same as image path but with .npy extension
    padding_path = str(image_object.get_step(cropping_step).path).replace(".tif", '.npy')
    padding = np.load(padding_path, allow_pickle=True)
    # incase images are at different resolutions
    padding =padding * pad_rescale
    print('padding loaded')
    img = np.pad(img, padding.astype(int), mode = mode)
    print('image padded')

    output = StepOutput(
        image_object.working_dir,
        step_name="center_pad",
        output_name=output_name,
        output_type="image",
        image_id=image_object.id,
    )
    output.save(img)
    print('image saved')
    return output

@flow(task_runner=create_task_runner(), log_prints=True)
def center_pad(
    image_object_paths,image_step, cropping_step, output_name, mode = 'constant', pad_rescale = 1
):
    image_objects = [ImageObject.parse_file(path) for path in image_object_paths]

    results = []
    for obj in image_objects:
        results.append(
            uncrop.submit(
                obj,
                output_name,
                image_step,
                cropping_step,
                mode=mode,
                pad_rescale = pad_rescale,
            )
        )
    step_outputs = [result.result() for result in results]
    for obj, step_output in zip(image_objects, step_outputs):
        obj.add_step_output(step_output)
        obj.save()