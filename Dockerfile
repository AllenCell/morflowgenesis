# syntax=docker/dockerfile:1.0.0-experimental
FROM docker-modeling-local.artifactory.corp.alleninstitute.org/mlops/prefect-gpu:v0.0.0

ENV GIT_SSH_COMMAND="ssh -vvv"
COPY . /tmp/morflowgenesis

RUN --mount=type=ssh,id=github \
    mamba install -y -n prefect \
      fortran-compiler blas-devel \
    && conda run -n prefect pip install numpy \
    && conda run -n prefect pip install -r /tmp/morflowgenesis/requirements.txt --no-cache-dir  -i https://artifactory.corp.alleninstitute.org/artifactory/api/pypi/pypi-virtual/simple \
    && conda run -n prefect pip install /tmp/morflowgenesis --no-cache-dir \
    && conda run -n prefect pip uninstall -y pynvml \
    && conda clean -afy \
    && conda run -n prefect pip cache purge

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
