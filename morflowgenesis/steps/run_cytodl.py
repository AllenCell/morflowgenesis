import os
import shutil
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

import mlflow
from cyto_dl.eval import evaluate
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict, read_write
from prefect import flow, task

from morflowgenesis.utils.image_object import ImageObject
from morflowgenesis.utils.step_output import StepOutput


def download_mlflow_model(
    run_id: str,
    save_path: str,
    checkpoint_path="checkpoints/val/loss/best.ckpt",
    tracking_uri: str = "https://mlflow.a100.int.allencell.org",
):
    if (save_path / checkpoint_path).exists():
        print("Checkpoint exists! Skipping download...")
        return save_path / checkpoint_path
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.artifacts.download_artifacts(
        run_id=run_id, tracking_uri=tracking_uri, artifact_path=checkpoint_path, dst_path=save_path
    )
    return save_path / checkpoint_path


@task
def generate_config(
    image_objects,
    input_step,
    config_path,
    overrides,
    run_id=None,
    checkpoint_path="checkpoints/val/loss/best.ckpt",
):
    working_dir = image_objects[0].working_dir

    mlflow_ckpt_path = None
    if run_id is not None:
        save_path = working_dir / run_id
        save_path.mkdir(exist_ok=True, parents=True)
        mlflow_ckpt_path = download_mlflow_model(run_id, save_path, checkpoint_path)

    # initialize config with overrides
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

        if mlflow_ckpt_path is not None:
            cfg["ckpt_path"] = mlflow_ckpt_path


        heads = list(cfg.model.task_heads.keys())
        if 'inference_heads' in cfg.model:
            heads = cfg.model.inference_heads

        save_dir = working_dir / "run_cytodl"

        # get input data path
        data_paths = [im.get_step(input_step).path for im in image_objects if not np.all([(save_dir/head/f'{im.id}.tif').exists() for head in heads])]

        # TODO make load/save path overrides work on default cytodl configs
        cfg["data"]["data"] = [{cfg.model.x_key: str(p)} for p in data_paths]
        cfg["model"]["save_dir"] = str(save_dir)
        HydraConfig.instance().set_config(cfg)
        OmegaConf.set_readonly(cfg.hydra, None)
    return cfg


@task(retries=3, retry_delay_seconds=[10, 60, 120])
def run_evaluate(cfg):
    return evaluate(cfg)


@flow(log_prints=True)
def run_cytodl(
    image_object_paths: List[Union[str, Path]],
    input_step: str,
    config_path: str,
    overrides: List = [],
    run_id: Optional[str] = None,
    checkpoint_path: Optional[str] = "checkpoints/val/loss/best.ckpt",
):
    """Wrapper function to run cytoDL on a list of image objects. Note that the output will be
    saved to `working_dir/run_cytodl/head_name` for each output task head of your model.

    Parameters
    ----------
    image_object_paths : List[str]
        List of paths to image objects to run
    input_step : str
        Name of step to use as input data
    config_path : str
        Path to base config file
    overrides : List, optional
        List of overrides to apply to config, by default []
    run_id : Optional[str], optional
        MLFlow run ID to download model from, by default None
    checkpoint_path : Optional[str], optional
        Path to checkpoint to download from MLFlow, by default "checkpoints/val/loss/best.ckpt"
    """
    image_objects = [ImageObject.parse_file(p) for p in image_object_paths]

    cfg = generate_config(
        image_objects,
        input_step,
        Path(config_path),
        overrides,
        run_id,
        checkpoint_path,
    )
    if len(cfg.data.data) == 0:
        return
    _, _, out = run_evaluate(cfg)
    for batch in out:
        for input_filename, output_dict in batch.items():
            for i in range(len(image_objects)):
                if input_filename == str(image_objects[i].get_step(input_step).path):
                    for head, save_path in output_dict.items():
                        output = StepOutput(
                            image_objects[0].working_dir,
                            step_name="run_cytodl",
                            output_name=head,
                            output_type="image",
                            image_id=image_objects[i].id,
                        )
                        image_objects[i].add_step_output(output)
                        image_objects[i].save()
                        shutil.move(str(save_path), str(output.path))
    # delete 'predict_images, 'test_images', train_images', and 'val_images' from the run_cytodl folder
    for folder in ["predict_images", "test_images", "train_images", "val_images"]:
        shutil.rmtree(Path(cfg.model.save_dir) / folder)   
