import os
import json
import base64

import dask
import distributed
from dask_kubernetes.classic import make_pod_spec
from prefect.task_runners import SequentialTaskRunner
from prefect_dask import DaskTaskRunner


def encode_dict_to_json_base64(input_dict):
    json_str = json.dumps(input_dict)
    json_bytes = json_str.encode('utf-8')
    return base64.b64encode(json_bytes).decode('utf-8')


def decode_base64_json_string(encoded_str):
    decoded_bytes = base64.b64decode(encoded_str)
    json_str = decoded_bytes.decode('utf-8')
    return json.loads(json_str)


def make_dask_cluster_kwargs(encoded_str):
    _dask_kwargs = {
        "cluster_class": "dask_kubernetes.KubeCluster",
        "cluster_kwargs": {
            "pod_template": {
                "memory_limit": "4Gi",
                "memory_request": "1Gi",
                "cpu_limit": "1000m",
                "cpu_request": "1000m",
                "env": {
                    "TZ": "UTC"
                }
            },
            "deploy_mode": "local",
            "n_workers": 5,
        }
    }

    _dask_kwargs["adapt_kwargs"] = {
        "minimum": _dask_kwargs["cluster_kwargs"]["n_workers"]
    }

    _user_provided_kwargs = decode_base64_json_string(encoded_str)
    _dask_kwargs.update(_user_provided_kwargs)
    _dask_kwargs["cluster_kwargs"]["pod_template"] = make_pod_spec(
        **_dask_kwargs["cluster_kwargs"]["pod_template"])

    if "maximum" not in _dask_kwargs["adapt_kwargs"]:
        _dask_kwargs["adapt_kwargs"]["maximum"] = (
            _dask_kwargs["adapt_kwargs"]["minimum"] * 2)

    return _dask_kwargs


def create_task_runner():
    dask.config.set({"distributed.diagnostics.nvml": False})

    if os.environ.get("DASK_CLUSTER") is not None:
        dask_kwargs = make_dask_cluster_kwargs(os.environ["DASK_CLUSTER"])
        return DaskTaskRunner(**dask_kwargs)

    return SequentialTaskRunner()
