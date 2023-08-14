from prefect import flow
import pandas as pd
from prefect.task_runners import ConcurrentTaskRunner, SequentialTaskRunner
from prefect.flows import Flow
from prefect.tasks import Task


@flow(task_runner=SequentialTaskRunner())
# Sequentially run steps to avoid overloading GPU/CPU resources. Eventually this should be concurrent.
def run_step(fn, step_type,  data):
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
