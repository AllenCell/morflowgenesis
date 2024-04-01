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


def find_timepoint_crop(
    image_object: ImageObject,
    image_step: str,
    pad: Union[int, List[int]] = 5,
    min_slices: int = 24,
    sigma_cutoff: Union[int, List[int]] = 2,
):
    """Crop to center of an image assuming a Gaussian-like profile along the z-axis./

    Parameters
    ----------
    image_object : ImageObject
        ImageObject to crop
    image_step : str
        Step name of image to crop
    pad : Union[int, List[int]]
        Number of slices to pad the crop. If list, first element is number of slices to pad below, second is number of slices to pad above.
    min_slices : int
        Minimum number of slices to crop to
    sigma_cutoff : Union[int, List[int]]
        Number of standard deviations to crop around the center of the Gaussian-like profile. If list, first element is number of standard deviations to crop below, second is number of standard deviations to crop above.
    """
    img = image_object.load_step(image_step)
    bottom_padding, top_padding = 0, 0
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
        bottom_padding = bottom_z
        top_padding = img.shape[0] - top_z
    print(f"{image_object.id} done")
    return bottom_padding, top_padding


def apply_crop(
    image_object: ImageObject,
    output_name: str,
    image_step: str,
    bottom_padding: int,
    top_padding: int,
):
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
    output.save(img[bottom_padding:-top_padding])
    image_object.add_step_output(output)
    image_object.save()


def center_crop(
    image_objects: List[ImageObject],
    tags: List[str],
    image_step: str,
    output_name: str,
    pad: Union[int, List[int]] = 5,
    min_slices: int = 24,
    sigma_cutoff: Union[int, List[int]] = 2,
):
    padding_path = image_objects[0].working_dir / "center_crop" / output_name / "padding.npy"
    if not padding_path.exists():
        _, results = parallelize_across_images(
            image_objects,
            find_timepoint_crop,
            tags=tags,
            image_step=image_step,
            pad=pad,
            min_slices=min_slices,
            sigma_cutoff=sigma_cutoff,
        )
        results = np.array(results)
        bottom_padding = np.max(results[:, 0])
        top_padding = np.max(results[:, 1])
        print("Per-timepoint padding complete, padding:", bottom_padding, top_padding)
        padding_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(padding_path,np.array([(bottom_padding, top_padding), (0, 0), (0, 0)]))
    else:
        padding = np.load(padding_path, allow_pickle=True)
        bottom_padding, top_padding = padding[0]
    print("Consensus padding is", bottom_padding, top_padding)


    parallelize_across_images(
        image_objects,
        apply_crop,
        tags=tags,
        image_step=image_step,
        output_name=output_name,
        bottom_padding=bottom_padding,
        top_padding=top_padding,
    )
    

def uncrop(
    image_object: ImageObject,
    output_name: str,
    image_step: str,
    mode: str = "constant",
    padding: List[int] = [0, 0, 0],
):
    """
    Uncrop image by padding it with zeros according to the padding from the cropping step.
    Parameters
    ----------
    image_object : ImageObject
        ImageObject to uncrop
    output_name : str
        Name of output
    image_step : str
        Step name of image to uncrop
    cropping_step : str
        Step name of cropping step
    mode : str
        Padding mode, see numpy.pad
    pad_rescale : float
        Rescale padding by this factor. Helpful if images are at different resolutions
    """
    output = StepOutput(
        image_object.working_dir,
        step_name="center_pad",
        output_name=output_name,
        output_type="image",
        image_id=image_object.id,
    )
    if not output.path.exists():
        img = image_object.load_step(image_step)
        print("image loaded", img.shape, img.dtype)
        img = np.pad(img, padding, mode=mode)
        print("image padded", img.shape)
        output.save(img)
    print("image saved")
    image_object.add_step_output(output)
    image_object.save()


def center_pad(
    image_objects: List[ImageObject],
    tags: List[str],
    image_step: str,
    cropping_step: str,
    output_name: str,
    mode: str = "constant",
    pad_rescale: float = 1.0,
):
    # crop path is same as image path but with .npy extension
    padding_path = str(image_objects[0].get_step(cropping_step).path.parent / "padding.npy")
    padding = np.load(padding_path, allow_pickle=True)
    # in case images are at different resolutions
    padding = np.round(padding * pad_rescale).astype(int)
    print("padding loaded", padding)

    parallelize_across_images(
        image_objects,
        uncrop,
        tags=tags,
        image_step=image_step,
        output_name=output_name,
        mode=mode,
        padding=padding,
    )
