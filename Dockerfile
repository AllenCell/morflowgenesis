# syntax=docker/dockerfile:1.0.0-experimental
FROM docker-modeling-local.artifactory.corp.alleninstitute.org/mlops/prefect-gpu:v0.0.0

ENV GIT_SSH_COMMAND="ssh -vvv"
COPY . /tmp/morflowgenesis

RUN --mount=type=ssh,id=github \
    mamba install -y -n prefect \
      fortran-compiler blas-devel \
    && pip install numpy \
    && pip install -r /tmp/morflowgenesis/requirements.txt --no-cache-dir \
    && pip install /tmp/morflowgenesis --no-cache-dir \
    && conda clean -afy \
    && prefect pip cache purge
