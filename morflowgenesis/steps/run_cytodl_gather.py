import os
import shutil
from pathlib import Path

from cyto_dl.eval import evaluate
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict, read_write
from prefect import flow, task
import pandas as pd

from morflowgenesis.utils import create_task_runner
from morflowgenesis.utils.step_output import StepOutput
from morflowgenesis.utils.image_object import ImageObject

@task 
def generate_config(image_objects, step_name, output_name,input_step,config_path, overrides):
    # get input data path
    data_paths = [im.get_step(input_step).path for im in image_objects]

    # initialize config with overrides
    config_path = Path(config_path)
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.2", config_dir=str(config_path.parent.resolve())):
        cfg = compose(config_name=config_path.name, return_hydra_config=True, overrides=overrides)
        output_dir = os.path.abspath(cfg.hydra.run.dir)
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        with read_write(cfg.hydra.runtime):
            with open_dict(cfg.hydra.runtime):
                cfg.hydra.runtime.output_dir = output_dir

        with read_write(cfg.hydra.job):
            with open_dict(cfg.hydra.job):
                cfg.hydra.job.num = 0
                cfg.hydra.job.id = 0

        # TODO make load/save path overrides work on default cytodl configs
        cfg["data"]["data"] = [{cfg.source_col: str(p)} for p in data_paths]
        save_dir = image_objects[0].working_dir / step_name / output_name
        cfg["model"]["save_dir"] = str(save_dir)
        HydraConfig.instance().set_config(cfg)
        OmegaConf.set_readonly(cfg.hydra, None)
    return cfg

@task(retries=3, retry_delay_seconds=[10,60,120])
def run_evaluate(cfg):
    return evaluate(cfg)

@flow(log_prints=True)
def run_cytodl(image_object_paths, step_name, output_name, input_step, config_path, overrides=[]):
    image_objects = [ImageObject.parse_file(p) for p in image_object_paths]

    cfg = generate_config(image_objects, step_name, output_name,input_step, config_path, overrides)
    _, _, out = run_evaluate(cfg)
    filename_map =pd.concat([pd.DataFrame(i) for i in out])
    if len(filename_map) != len(image_objects):
        breakpoint()

    for row in filename_map.itertuples():
        for i in range(len(image_objects)):
            if row.input  == str(image_objects[i].get_step(input_step).path):
                output =  StepOutput(
                    image_objects[0].working_dir,
                    step_name=step_name,
                    output_name=output_name,
                    output_type="image",
                    image_id=image_objects[i].id,
                ) 
                image_objects[i].add_step_output(output)
                image_objects[i].save()
                try:
                    shutil.move(str(row.output), str(output.path))
                finally:
                    break




