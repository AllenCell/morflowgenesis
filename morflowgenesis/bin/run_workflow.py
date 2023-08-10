import hydra
from omegaconf import DictConfig
from prefect import flow, get_run_logger, task
from prefect.task_runners import ConcurrentTaskRunner, SequentialTaskRunner
from morflowgenesis.bin.run_step import run_step
from pathlib import Path
import pickle


@task
def load_image_objects(obj_path):
    with open(obj_path, 'rb') as f:
        return pickle.load(f)
    
@flow(task_runner=ConcurrentTaskRunner())
def get_workflow_state(cfg):
    logger= get_run_logger()
    existing_objects= []
    img_object_dir =Path(cfg['working_dir'])/'_ImageObjectStore'
    for obj_path in img_object_dir.glob('*.pkl'):
        existing_objects.append(load_image_objects.submit(obj_path))
    logger.info(f'Loaded {len(existing_objects)} image objects')
    return [x.result() for x in existing_objects]


@flow(task_runner=SequentialTaskRunner())
def WorkflowRunner_object(cfg):
    prev_output = get_workflow_state(cfg)
    for step_name, step_meta in cfg['steps'].items():
        step_fn = step_meta['function']
        step_type = step_meta['step_type']
        step = hydra.utils.instantiate(step_fn)
        out = run_step(step, step_type, prev_output)
        prev_output = out

@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    WorkflowRunner_object(cfg)

if __name__ == "__main__":
    main()