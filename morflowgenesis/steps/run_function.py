from typing import Callable, Dict, List, Optional

from hydra._internal.utils import _locate

from morflowgenesis.utils import (
    ImageObject,
    StepOutput,
    parallelize_across_images,
    to_list,
)


def apply_function(image_object, input_step, output_name, ch, function, function_args):
    # extract inputs
    data = image_object.load_step(input_step)
    if ch is not None:
        data = data[ch]

    # initialize and apply function
    function = _locate(function)
    applied = function(data, **function_args)

    # add result to image object
    output = StepOutput(
        working_dir=image_object.working_dir,
        step_name="array_to_array",
        output_name=f"{output_name}/{input_step}",
        output_type="image",
        image_id=image_object.id,
    )
    output.save(applied)
    image_object.add_step_output(output)
    image_object.save()


def array_to_array(
    image_objects: List[ImageObject],
    output_name: str,
    input_steps: List[str],
    function: Callable,
    ch: Optional[int] = None,
    function_args: Dict[str, str] = {},
):
    """Apply a function to an array and save the result to the image object
    Parameters
    ----------
    image_objects : List[ImageObject]
        List of ImageObjects to run function
    output_name : str
        Name of output. The input step name will be appended to this name in the format `output_name/input_step`
    input_steps : List[str]
        Step names of input images
    function: Callable
        Function to apply to the array
    ch: Optional[int]
        Channel to apply the function to. By default, applies to entire image
    function_args: Dict[str,str]
        Arguments to pass to the function
    """

    input_steps = to_list(input_steps)

    for step in input_steps:
        parallelize_across_images(
            image_objects,
            apply_function,
            input_step=step,
            output_name=output_name,
            ch=ch,
            function=function,
            function_args=function_args,
        )
