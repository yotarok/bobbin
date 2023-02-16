****************************************
Bobbin
****************************************

Bobbin is a complemental tool for building training loop with Flax.
This is not intended to be a full-scale framework, rather it is something you
can partally import and use (for typically small experiments that don't require
frameworks.)

Features
^^^^^^^^

- TrainTask: Simple abstraction of training task, and provides training step
  function.
- EvalTask: Simple abstraction of evaluation task and provides multi-host,
  multi-device parallelization of evaluation tasks.
- CronTab: A table that can keep track of periodically executed actions.
- TensorBoard: Some mechanism to publish training intermediates, and other
  tools wrapping `flax.metrics.tensorboard` API.
- Other utility functions
  - var_util: Some typical pytree manipulation
  - pmap_util: Some easy accessor for pmap
  - pytypes: Some type definitions for self-documenting code.

Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install bobbin

Basic usage
^^^^^^^^^^^

.. toctree::
   :hidden:
   :caption: Basic Usage
   :maxdepth: 1

   pmap_util

   var_util


API documentation
^^^^^^^^^^^^^^^^^

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API

   api
