from omegaconf import ListConfig
from prefect.tasks import Task
from prefect.flows import Flow

def submit(task_function, tags = [], **kwargs):
    task = Task(task_function, name=task_function.__name__, tags=tags, log_prints=True)
    return task.submit(**kwargs)


def parallelize_across_images(data, task_function, tags = [], data_name='image_object', **kwargs):
    """
    data is list of image objects
    results is list of step outputs, one per image object
    """
    results = []
    for d in data:
        kwargs.update({data_name: d})
        results.append(submit(task_function, tags = tags, **kwargs))
    results = [r.result() for r in results]
    for d, r in zip(data, results):
        d.add_step_output(r)
        d.save()
    return data



def parallelize_across_objects(data, task_function, object_extraction_fn, combine_results_fn, tags = [], **kwargs):
    """
    data is list of image objects
    """
    for d in data:
        results = []
        objects = object_extraction_fn(d)
        for i in objects:
            kwargs.update({'image_object': d})
            kwargs.update(i)        
            results.append(submit(task_function, tags=tags, **kwargs))
        comb = combine_results_fn(image_object = d, results = [r.result() for r in results])
        d.add_step_output(comb)
        d.save()
    return data

def run_flow(flow_function, task_runner, run_type, tags, **kwargs):
    flow = Flow(flow_function, task_runner = task_runner)
    kwargs.update({'run_type': run_type, "tags": tags})
    flow._run(**kwargs)


def to_list(x):
    if isinstance(x, (list, ListConfig)):
        return x
    else:
        return [x]