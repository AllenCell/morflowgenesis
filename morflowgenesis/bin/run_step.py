import pandas as pd
from prefect.deployments.deployments import Deployment, run_deployment

# TODO make csv/list steps non-blocking
# Sequentially run steps to avoid overloading GPU/CPU resources. Eventually this should be concurrent.
def run_step(fn, step_name, step_type,  data):
    breakpoint()
    results = []
    if step_type == 'csv':
        assert isinstance(data, pd.DataFrame), 'input data must be csv for csv steps, got {}'.format(type(data))
        for row in data.itertuples():
            dep = Deployment(step_name, parameters = {'data': row})
            dep.build_from_flow(fn, name=str(row))
            dep_id = dep.apply()
            results.append(run_deployment(step_name))
    elif step_type == 'list':
        for d in data:
            steps.build_from_flow(fn)
            #d
    elif step_type == 'string' or step_type == 'gather':
        return fn(data, **fn.keywords)
    elif step_type == 'none':
        dep = Deployment.build_from_flow(flow=fn.func, name=str(step_name),)
        dep_id = dep.apply()
        data = {'data':data}
        data.update(fn.keywords)
        results.append(run_deployment(dep_id,parameters = data))
        # return fn(data, **fn.keywords)

    return results