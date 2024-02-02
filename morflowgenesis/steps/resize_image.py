from skimage.transform import rescale as sk_rescale
from skimage.transform import resize as sk_resize

from morflowgenesis.utils import (
    ImageObject,
    StepOutput,
    to_list,
    parallelize_across_images,
)

def run_resize(image_object, output_name, input_step, output_shape=None, scale=None, order=0):
    # image resizing
    img = image_object.load_step(input_step)
    input_dtype = img.dtype
    if output_shape is not None:
        img = sk_resize(img, output_shape, order=order, preserve_range=True, anti_aliasing=False)
    elif scale is not None:
        img = sk_rescale(img, scale, order=order, preserve_range=True, anti_aliasing=False)
    # add result to image object
    output = StepOutput(
        image_object.working_dir,
        step_name="resize",
        output_name=f"{output_name}_{input_step}",
        output_type="image",
        image_id=image_object.id,
    )
    output.save(img.astype(input_dtype))
    return output

def resize(image_object_paths, tags = [], output_name, input_steps, output_shape=None, scale=None, order=0, run_type=None):
    """
    Resize images to a specified shape or scale with a specified order of interpolation.
    """
    input_steps = to_list(input_steps)

    image_objects = [ImageObject.parse_file(path) for path in image_object_paths]
    for step in input_steps:
        parallelize_across_images(
            image_objects,
            run_resize,
            tags=tags,
            output_name=output_name,
            input_steps=step,
            output_shape=output_shape,
            scale=scale,
            order=order,
            run_within_object=True,
        )