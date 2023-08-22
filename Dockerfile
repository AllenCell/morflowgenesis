# syntax=docker/dockerfile:1.0.0-experimental
FROM docker-modeling-local.artifactory.corp.alleninstitute.org/mlops/prefect-gpu:latest

ENV GIT_SSH_COMMAND="ssh -vvv"
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=ssh,id=github \
    mamba install -y -n base \
      fortran-compiler blas-devel \
    && pip install -r /tmp/requirements.txt --no-cache-dir \
    && conda clean -afy \
    && pip cache purge
