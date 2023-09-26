from hydra._internal.utils import _locate
from prefect import flow, task

from morflowgenesis.utils.step_output import StepOutput
from morflowgenesis.utils.image_object import ImageObject



@task
def load(image_object_path, input_step):
    image_object = ImageObject.parse_file(image_object_path)
    return image_object, image_object.load_step(input_step)

@task
def run(function, function_args, data):
    function= _locate(function)
    return function(data, **function_args)


@flow(log_prints=True)
def array_to_array(
    image_object_path, step_name, output_name, input_step, function, function_args={}
):
    
    image_object, data = load(image_object_path, input_step)
    results = run(function, function_args, data)

    output = StepOutput(
        working_dir = image_object.working_dir,
        step_name = step_name,
        output_name = output_name, 
        output_type = 'image',
        image_id = image_object.id
    )
    output.save(results)
    image_object.add_step_output(output)
    image_object.save()
