from hydra._internal.utils import _locate

from morflowgenesis.utils import (
    ImageObject,
    StepOutput,
    parallelize_across_images,
    to_list,
)


def apply_function(image_object, input_step, output_name, ch, function, function_args):
    data = image_object.load_step(input_step)
    if ch is not None:
        data = data[ch]
    function = _locate(function)

    applied = function(data, **function_args)
    output = StepOutput(
        working_dir=image_object.working_dir,
        step_name="array_to_array",
        output_name=f"{output_name}/{input_step}",
        output_type="image",
        image_id=image_object.id,
    )
    output.save(applied)
    return output


def array_to_array(
    image_objects, output_name, input_steps, function, ch=None, function_args={}
):
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
