from numbers import Number
from typing import List, Union

import numpy as np
from scipy.optimize import curve_fit

from morflowgenesis.utils import ImageObject, StepOutput, parallelize_across_images


def gaussian(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def extract_values(x):
    if isinstance(x, list):
        return x[0], x[1]
    elif isinstance(x, Number):
        return x, x
    else:
        raise ValueError("Input must be tuple, list, or number")


def crop(
    image_object: ImageObject,
    output_name: str,
    image_step: str,
    pad: Union[int, List[int]] = 5,
    min_slices: int = 24,
    sigma_cutoff: Union[int, List[int]] = 2,
):
    """Crop to center of an image assuming a Gaussian-like profile along the z-axis."""
    # add result to image object
    output = StepOutput(
        image_object.working_dir,
        step_name="center_crop",
        output_name=output_name,
        output_type="image",
        image_id=image_object.id,
    )
    if output.path.exists():
        return output
    img = image_object.load_step(image_step)
    padding = np.array([(0, 0), (0, 0), (0, 0)])
    if len(img.shape) != 3:
        raise ValueError("Image must be 3D")

    if img.shape[0] < min_slices:
        raise ValueError(f"Image must have at least {min_slices}, got {img.shape[0]}")
    elif img.shape[0] > min_slices:
        # only crop if image != min_slices
        # minmax normalize z-profile of standard deviation and fit gaussian
        z_prof = np.std(img, axis=(1, 2))
        z_prof = (z_prof - np.min(z_prof)) / (np.max(z_prof) - np.min(z_prof))
        popt, _ = curve_fit(gaussian, np.arange(len(z_prof)), z_prof, p0=[1, len(z_prof) / 2, 0.3])
        _, center, sigma = popt

        sigma_cutoff_low, sigma_cutoff_high = extract_values(sigma_cutoff)
        pad_low, pad_high = extract_values(pad)

        # ensure crop is within bounds of image
        bottom_z = max(0, int(center - sigma * sigma_cutoff_low) - pad_low)
        top_z = min(img.shape[0], int(center + sigma * sigma_cutoff_high) + pad_high)

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
        padding[0] = [bottom_z, img.shape[0] - top_z]
        img = img[bottom_z:top_z]
    output.save(img)
    np.save(
        image_object.working_dir / "center_crop" / output_name / f"{image_object.id}.npy", padding
    )
    return output


def uncrop(image_object, output_name, image_step, cropping_step, mode="constant", pad_rescale=1):
    output = StepOutput(
        image_object.working_dir,
        step_name="center_pad",
        output_name=output_name,
        output_type="image",
        image_id=image_object.id,
    )
    if not output.path.exists():
        img = image_object.load_step(image_step)
        print("image loaded")
        # crop path is same as image path but with .npy extension
        padding_path = str(image_object.get_step(cropping_step).path).replace(".tif", ".npy")
        padding = np.load(padding_path, allow_pickle=True)
        # in case images are at different resolutions
        padding = padding * pad_rescale
        print("padding loaded")
        img = np.pad(img, padding.astype(int), mode=mode)
        print("image padded")
        output.save(img)
    print("image saved")
    return output


def center_crop(
    image_objects,
    tags,
    image_step,
    output_name,
    pad=5,
    min_slices=24,
    sigma_cutoff=2,
):
    parallelize_across_images(
        image_objects,
        crop,
        tags=tags,
        image_step=image_step,
        output_name=output_name,
        pad=pad,
        min_slices=min_slices,
        sigma_cutoff=sigma_cutoff,
    )


def center_pad(
    image_objects,
    tags,
    image_step,
    cropping_step,
    output_name,
    mode="constant",
    pad_rescale=1,
):
    parallelize_across_images(
        image_objects,
        uncrop,
        tags=tags,
        image_step=image_step,
        cropping_step=cropping_step,
        output_name=output_name,
        mode=mode,
        pad_rescale=pad_rescale,
    )
