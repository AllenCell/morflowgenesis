from omegaconf import ListConfig, DictConfig, OmegaConf

def submit(task_function, as_task, **kwargs):
    if as_task:
        return task_function.submit(**kwargs)
    else:
        return task_function.fn(**kwargs)

def to_list(x):
    if isinstance(x, (list, ListConfig)):
        return x
    else:
        return [x]