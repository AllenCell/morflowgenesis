import os
import shutil
from pathlib import Path

from cyto_dl.eval import evaluate
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict, read_write
from prefect import flow

from morflowgenesis.utils import create_task_runner
from morflowgenesis.utils.step_output import StepOutput
from morflowgenesis.utils.image_object import ImageObject



@flow(task_runner=create_task_runner(), log_prints=True)
def run_cytodl(image_object, step_name, output_name, input_step, config_path, overrides=[]):
    image_object = ImageObject.parse_file(image_object)
    # skip if already run
    if image_object.step_is_run(f"{step_name}_{output_name}"):
        return image_object

    # get input data path
    prev_step_output = image_object.get_step(input_step)
    data_path = prev_step_output.path

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
        cfg["data"]["data"] = [{cfg.source_col: str(data_path)}]
        save_dir = image_object.working_dir / step_name / output_name / image_object.id
        cfg["model"]["save_dir"] = str(save_dir)

        HydraConfig.instance().set_config(cfg)
        OmegaConf.set_readonly(cfg.hydra, None)
        evaluate(cfg)

    # find where cytodl saves out image
    out_image_path = list((save_dir / "predict_images").glob("*"))[0]
    output = StepOutput(
        image_object.working_dir,
        step_name=step_name,
        output_name=output_name,
        output_type="image",
        image_id=image_object.id,
    )
    # move it to expected path of output step
    shutil.move(str(out_image_path), str(output.path))
    # delete cytodl-generated predict_image/test_image etc. folders
    shutil.rmtree(save_dir)

    image_object.add_step_output(output)
    image_object.save()
