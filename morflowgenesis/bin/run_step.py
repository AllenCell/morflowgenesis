import pandas as pd

# TODO make csv/list steps non-blocking
# Sequentially run steps to avoid overloading GPU/CPU resources. Eventually this should be concurrent.
def run_step(fn, step_type,  data):
    issues = []
    if step_type == 'csv':
        assert isinstance(data, pd.DataFrame), 'input data must be csv for csv steps, got {}'.format(type(data))
        for row in data.itertuples():
            issues.append(fn(row))
    elif step_type == 'list':
        for d in data:
            issues.append(fn(d))
    elif step_type == 'string' or step_type == 'gather':
        return fn(data, **fn.keywords)
    elif step_type == 'none':
        return fn(data, **fn.keywords)

    return issues