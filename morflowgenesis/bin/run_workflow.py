import hydra
from omegaconf import DictConfig
from morflowgenesis.bin.run_step import run_step
from pathlib import Path
from omegaconf import OmegaConf
from morflowgenesis.steps.load_workflow_state import get_workflow_state

def save_workflow_config(working_dir, cfg):
    with open(working_dir/'workflow_config.yaml', 'w') as f:
        OmegaConf.save(config= cfg, f = f)

def WorkflowRunner(cfg):
    """
    Sequentially run config-specified steps, starting with the previous workflow state
    and passing output from step n-1 as input to step n
    """
    working_dir = Path(cfg['working_dir'])
    working_dir.mkdir(exist_ok=True, parents=True)
    save_workflow_config(working_dir, cfg)

    prev_output = get_workflow_state(cfg)
    for step_name, step_meta in cfg['steps'].items():
        step_fn = step_meta['function']
        step_type = step_meta['step_type']
        step = hydra.utils.instantiate(step_fn)
        out = run_step(step,step_name, step_type, prev_output)
        prev_output = out

@hydra.main(version_base="1.3", config_path="../configs/workflow", config_name="config.yaml")
# default config is morflowgenesis/configs/workflow/config.yaml
def main(cfg: DictConfig):
    WorkflowRunner(cfg)

if __name__ == "__main__":
    main()