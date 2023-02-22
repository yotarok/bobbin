Training
========

.. currentmodule:: bobbin

.. autosummary::

    BaseTrainTask
    TrainTask
    TrainState

Train tasks
~~~~~~~~~~~

.. autoclass:: BaseTrainTask
    :members:

.. autoclass:: TrainTask
    :members:

Train state
~~~~~~~~~~~

.. autoclass:: TrainState
    :members:

Evaluation
==========

.. autosummary::

    EvalResults
    SampledSet
    EvalTask

Evaluation results
~~~~~~~~~~~~~~~~~~

.. autoclass:: EvalResults
    :members:

.. autoclass:: SampledSet
    :members:

Evaluation tasks
~~~~~~~~~~~~~~~~

.. autoclass:: EvalTask
    :members:

Crontab
=======

.. autosummary::
    CronTab

.. autoclass:: CronTab
    :members:

TensorBoard
===========

.. autosummary::
    NullSummaryWriter
    ImageSummary
    MplImageSummary
    MultiDirectorySummaryWriter
    ScalarSummary
    ThreadedSummaryWriter
    publish_train_intermediates
    publish_trainer_env_info

Summary writers
~~~~~~~~~~~~~~~

.. autoclass:: NullSummaryWriter

.. autoclass:: MultiDirectorySummaryWriter

.. autoclass:: ThreadedSummaryWriter

Summary variable wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ImageSummary

.. autoclass:: MplImageSummary

.. autoclass:: ScalarSummary

Publish functions
~~~~~~~~~~~~~~~~~

.. autofunction:: publish_train_intermediates
.. autofunction:: publish_trainer_env_info


Pmap utils
==========

.. autosummary::

    tpmap
    unshard
    gather_from_jax_processes
    assert_replica_integrity


.. autofunction:: tpmap
.. autofunction:: unshard
.. autofunction:: gather_from_jax_processes
.. autofunction:: assert_replica_integrity

Var utils
=========

.. autosummary::

    flatten_with_paths
    nested_vars_to_paths
    dump_pytree_json
    parse_pytree_json
    read_pytree_json_file
    write_pytree_json_file
    summarize_shape
    total_dimensionality

Path
~~~~

.. autofunction:: flatten_with_paths
.. autofunction:: nested_vars_to_paths

JSON I/O
~~~~~~~~

.. autofunction:: dump_pytree_json
.. autofunction:: parse_pytree_json
.. autofunction:: read_pytree_json_file
.. autofunction:: write_pytree_json_file

Summarization
~~~~~~~~~~~~~

.. autofunction:: summarize_shape
.. autofunction:: total_dimensionality


