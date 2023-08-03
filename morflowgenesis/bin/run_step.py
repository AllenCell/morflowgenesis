from prefect import flow, task, get_run_logger
import pandas as pd
from prefect.task_runners import ConcurrentTaskRunner, SequentialTaskRunner
from prefect_dask import DaskTaskRunner
from prefect.flows import Flow
from prefect.tasks import Task
import dask

# @flow(task_runner=ConcurrentTaskRunner())
# def run_step(fn, step_type,  data, **kwargs):
#     # logger = get_run_logger()
#     issues = []

#     if step_type == 'csv':
#         assert isinstance(data, pd.DataFrame), 'input data must be csv for csv steps, got {}'.format(type(data))
#         for row in data.itertuples():
#             if isinstance(fn.func, Flow):
#                 issues.append(fn(row))
#             elif isinstance(fn.func, Task):
#                 issues.append(fn.func.submit(row, **fn.keywords))
#     elif step_type == 'list':
#         for d in data:
#             # issues.append(fn.func.submit(d, **fn.keywords))
#             issues.append(fn(d))
#     elif step_type == 'string':
#         return fn(data, **fn.keywords)
#     # return issues
#     return[p.result() for p in issues ]



@flow(task_runner=SequentialTaskRunner())
def run_step(fn, step_type,  data):
    # with dask.annotate(resource = {'GPU':1}):
    # logger = get_run_logger()
    issues = []
    if step_type == 'csv':
        assert isinstance(data, pd.DataFrame), 'input data must be csv for csv steps, got {}'.format(type(data))
        for row in data.itertuples():
            if isinstance(fn.func, Flow):
                issues.append(fn(row))
            elif isinstance(fn.func, Task):
                issues.append(fn.func.submit(row, **fn.keywords))
    elif step_type == 'list':
        for d in data:
            if isinstance(fn.func, Flow):
                issues.append(fn(d))
            elif isinstance(fn.func, Task):
                issues.append(fn.func.submit(d, **fn.keywords))
    elif step_type == 'string' or step_type == 'gather':
        return fn(data, **fn.keywords)
    elif step_type == 'none':
        return fn(data, **fn.keywords)

    # return issues
    try:
        return[p.result() for p in issues ]
    except:
        return issues
