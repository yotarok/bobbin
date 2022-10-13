# Bobbin

A small tool for making neural net training loop with flax.linen.

This library is designed to be a set of small tools and not to be a full-blown
framework for deep learning.  Users should keep freedom not to use some parts
of this library and the rest of the library should still be able to serve the
users.

Currently, this library contains the following components:

- `cron`:
  Tools for defining and performing periodical actions in the training-loop.
- `evaluation`:
  Helpers for performing evaluation and handling metrics.
- `pmap_utils`:
  Tools for helping the use of `jax.pmap`.
- `pytypes`:
  Type definitions for better type annotations.
- `tensorboard`:
  Helper functions for publishing training state and evaluation summaries.
- `training`:
  Helpers for writing training-loops for Flax models.
- `var_utils`:
  Utility for accessing variable collections of Flax models.


## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.

## Disclaimer

This library is still in a very early phase of development, and therefore,
its API can often be subject to destructive change.

This project is not an official Google project. It is not supported by
Google and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.
