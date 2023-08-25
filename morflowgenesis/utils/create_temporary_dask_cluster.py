import os

from dask_kubernetes.classic import make_pod_spec
from prefect.task_runners import SequentialTaskRunner
from prefect_dask import DaskTaskRunner


def create_task_runner():
    if os.environ.get("IMAGE") is not None:
        cluster_kwargs = {
            "pod_template": make_pod_spec(
                **{
                    "image": os.environ["IMAGE"],
                    "memory_limit": os.getenv("MEMORY_LIMIT", "4G"),
                    "memory_request": os.getenv("MEMORY_REQUEST", "1G"),
                    "cpu_limit": os.getenv("CPU_LIMIT", "1000m"),
                    "cpu_request": os.getenv("CPU_REQUEST", "1000m"),
                    "env": {"EXTRA_PIP_PACKAGES": "."},
                }
            ),
            "deploy_mode": "local",
            "n_workers": os.environ.get("NUM_DASK_WORKERS", 5),
        }
        adapt_kwargs = {
            "minimum": cluster_kwargs["n_workers"],
            "maximum": os.environ.get("MAX_DASK_WORKERS", 2 * cluster_kwargs["n_workers"]),
        }
        return DaskTaskRunner(
            cluster_class="dask_kubernetes.KubeCluster",
            cluster_kwargs=cluster_kwargs,
            adapt_kwargs=adapt_kwargs,
        )
    return SequentialTaskRunner()
