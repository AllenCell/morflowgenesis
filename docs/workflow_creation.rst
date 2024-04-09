.. highlight:: shell

============
Creating your own workflows
============

Prerequisites
--------------
Morflowgenesis uses `hydra`_ and `omegaconf`_ to allow flexible configuration of workflows involving `CytoDL`_ models. Familiarity with each of these tools is highly recommended for the easiest morflowgenesis experience.

.. _hydra: https://hydra.cc/
.. _omegaconf: https://omegaconf.readthedocs.io/en/2.3_branch/
.. _CytoDL: https://github.com/AllenCellModeling/cyto-dl

Overview
--------------
Morflowgenesis workflows create a DAG of steps to process data. Connections between steps are specified by the input and output arguments from each step. For example, a step that splits images into channels might be followed by a step that segments the channels. If the image splitting step is called `split_image` and has the output name `test`, the segmentation step can reference it by taking as its input step `split_image/test`.

Config Structure
--------------

The default config
^^^^^^^^^^^^^^

Morflowgenesis uses a hierarchical configuration structure to allow for easy customization of workflows. Running a workflow with no arguments will use the `default configuration`_. The `defaults` section references the other parameters that will be incorporated into the config file.

.. _default configuration: ../morflowgenesis/configs/workflow.yaml

The paths config
^^^^^^^^^^^^^

For example, the `paths` default includes a `paths` key in the config that references a `root_dir` and `model_dir`. These can be referred to in any of the configs using the OmegaConf interpolation syntax `${paths.root_dir}` and `${paths.model_dir}`.

The workflow config
^^^^^^^^^^^^^^

This default workflow, however, has no steps. To add steps, you can specify a `workflow` config with
.. code-block:: console

    $ python morflowgenesis/bin/run_workflow config=workflow_name.yaml

This will replace the `null` workflow config with the config in `configs/workflow_name.yaml`. The workflow config consists of a list of `steps` (included in the order specified by the `defaults` key at the config's top) and workflow-level parameters. Let's dissect the top of this workflow config:

.. code-block:: yaml

    defaults:
    - split_image@steps: ../../steps/split_image

The key `split_image@steps` imports the default `step` config at `steps/split_image` and includes it under the key `workflow.steps.split_image`. We can then override its parameters from within the workflow config as follows:

.. code-block:: yaml

    steps:
        split_image:
            image_path: path/to/images

This will update the default image path with the path to your images.

**Note**: including two of the same step in a workflow requires a different syntax:

.. code-block:: yaml

    defaults:
    - split_image@steps.split1: ../../steps/split_image
    - split_image@steps.split2: ../../steps/split_image

.. code-block:: yaml

    steps:
        split1:
            split_image:
                image_path: path/to/images1
        split2:
            split_image:
                image_path: path/to/images2

**Note**: Not all arguments can be overridden from the workflow config. Changing the *type* of `task_runner` for a step requires modifying the `step` config itself.

The step config
^^^^^^^^^^^^
Step configs specify default arguments for a given step. This must include a key that matches the name of the step, a `function` key that specifies the function location, and any arguments to that function. Task runners are a frequent addition to function arguments.

The task runner config
^^^^^^^^^^^^^^
Two task runners are provided - one for running `cpu`_ jobs in parallel and one for running `gpu`_ jobs in parallel. Step-specific arguments can be overridden from the `step` config. The GPU task runner should be updated with your system's gpu information (GPU memory limits, names, etc.). If no task runner is provided, the step will run using a naive parallelization strategy with prefect's `SequntialTaskRunner`_. To prevent overwhelming your resources, task runners are constrained by a `task_limit` argument, which corresponds to prefect's `concurrency-limit`.

If runs are cancelled before their concurrency limits can be cleaned up, you may need to manually clean up the unused concurrency limits. You can do this with the following command (WARNING this will delete the first 200 concurrency limits)

.. code-block:: console
    $ prefect concurrency-limit ls --limit 200 |  awk '{print $2}' | while read arg; do prefect concurrency-limit delete $arg; done

.. _SequntialTaskRunner: https://docs.prefect.io/latest/concepts/task-runners/
.. _cpu: ../morflowgenesis/configs/task_runners/dask.yaml
.. _gpu: ../morflowgenesis/configs/task_runners/dask_gpu.yaml

The params config
^^^^^^^^^^^^
With your worklow established, it's time to process! The `params` config is where you specify the parameters for running your workflow across different types of data, using different versions of the same model, or using different processing parameters. One `.yaml` file should be created for each parameter combination. For example, sweeping across images and models would require a `params` config for each image/model combination that you want to try. These parameters can be accessed within the workflow config as `${params.image}` and `{params.model}`. Running across all parameters can be done with the `--multirun` or `-m` flag.

.. code-block:: console

    $ python morflowgenesis/bin/run_workflow config=workflow_name.yaml params=image1_model1.yaml,image1_model2.yaml,image2_model1.yaml,image2_model2.yaml -m


.. _Github repo: https://github.com/AllenCell/morflowgenesis
.. _tarball: https://github.com/AllenCell/morflowgenesis/tarball/master
