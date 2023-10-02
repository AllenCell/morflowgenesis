import mlflow
import fire
import tempfile

from pathlib import Path
from omegaconf import OmegaConf

def main(run_id, save_path):
    tracking_uri = "https://mlflow.a100.int.allencell.org"

    mlflow.set_tracking_uri(tracking_uri)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config/train.yaml"

        mlflow.artifacts.download_artifacts(
            run_id=run_id, tracking_uri=tracking_uri, artifact_path="config/train.yaml",
            dst_path=tmp_dir
        )
        config = OmegaConf.load(config_path)
        config['model']['save_dir'] = None
        config['test']= False
        config['callbacks'] = None
        config['paths']={'output_dir':"${model.save_dir}"}

        base_morflow_data= {
            '_target_': "cyto_dl.datamodules.data_dict.make_data_dict_dataloader",
            'data':None,
            "transforms":[],
            "num_workers": 1,
            "batch_size": 1,
            "pin_memory": True,
            "persistent_workers": False,
        }

        transforms= config['data']['transforms']['train']['transforms']
        for t in transforms:
            if 'crop' not in t['_target_'].lower() and 'rand' not in t['_target_'].lower():
                base_morflow_data['transforms'].append(t)
            else:
                break
        config['data'] = base_morflow_data

    with open(save_path, "w") as f:
        OmegaConf.save(config=config, f=f)

if __name__ == '__main__':
    fire.Fire(main)
