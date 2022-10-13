# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Any, Callable, List, Mapping, Optional, Sequence, Union

import chex
import flax
import jax

from .pytypes import Backend
from .pytypes import Device

_ArrayTree = chex.ArrayTree


def pmap_shard_tree(x: _ArrayTree, ndevices: int) -> _ArrayTree:
    """Reshapes `(N, ...)`-arrays in the tree into `(ndevices, N // ndevices, ...)`."""

    def reshape_for_sharding(x):
        size, *rest_dims = x.shape
        if size % ndevices != 0:
            raise ValueError(
                "Size in leading axis of `x` must be a multiple "
                f"of # of devices ({ndevices})."
            )
        return x.reshape((ndevices, size // ndevices) + tuple(rest_dims))

    return jax.tree_util.tree_map(reshape_for_sharding, x)


class ArgType(metaclass=abc.ABCMeta):
    """Description for arguments passed to function obtained by `wrapped_pmap`."""

    @abc.abstractmethod
    def transform(self, x: _ArrayTree, devices: Sequence[Device]):
        ...

    def is_static(self):
        return False


class StaticArg(ArgType):
    """`ArgType` for static args."""

    def transform(self, x: _ArrayTree, devices: Sequence[Device]):
        return x

    def is_static(self):
        return True


class ThruArg(ArgType):
    """`ArgType` for preprocessed args. No wrapper applied to args with this type."""

    def transform(self, x: _ArrayTree, devices: Sequence[Device]):
        return x


class BroadcastArg(ArgType):
    """`ArgType` for args to be broadcasted."""

    def transform(self, x: _ArrayTree, devices: Sequence[Device]):
        return flax.jax_utils.replicate(x, devices)


class ShardArg(ArgType):
    """`ArgType` for args to be sharded and distributed over devices."""

    def transform(self, x: _ArrayTree, devices: Sequence[Device]):
        return pmap_shard_tree(x, len(devices))


class RngArg(ArgType):
    """`ArgType` for RNGs. The RNG args are split and distributed."""

    def transform(self, x: _ArrayTree, devices: Sequence[Device]):
        ndevices = len(devices)
        rngs = list(jax.random.split(x, num=ndevices))
        rngs = jax.device_put_sharded(rngs, devices)
        return rngs


def _id_fn(x: Any) -> Any:
    return x


_ARGTYPE_STRING_ALIASES = {
    "static": StaticArg(),
    "thru": ThruArg(),
    "broadcast": BroadcastArg(),
    "shard": ShardArg(),
    "rng": RngArg(),
}


def resolve_argtype_str(ty: Union[str, ArgType]) -> ArgType:
    """Converts string representation of `ArgType` to an instance."""
    if isinstance(ty, str):
        name = ty.lower()
        if name not in _ARGTYPE_STRING_ALIASES:
            raise ValueError(f'Unknown argtype name "{name}" is specified.')
        return _ARGTYPE_STRING_ALIASES[name]
    else:
        return ty


def wrapped_pmap(
    f: Callable,
    axis_name: str,
    argtypes: Sequence[Union[str, ArgType]],
    kwargtypes: Optional[Mapping[str, Union[str, ArgType]]] = None,
    *,
    devices: Optional[Device] = None,
    backend: Optional[Backend] = None,
    wrap_return: Optional[Callable[[_ArrayTree], _ArrayTree]] = None,
    **kwargs,
) -> Callable:
    """Wrap a Jax function so it can be transparently performed on multiple devices.

    This wraps the function `f` with `jax.pmap` with applying argument-and-return
    wrappers for ensuring API compatibility with the original function.

    Args:
        f: Function to be wrapped.
        axis_name: axis_name used in `jax.pmap`.
        argtypes: `ArgType` instances or string-representation of those describing
            distribution strategies of arguments.
        kwargtypes: `argtypes` for keyword arguments.
        devices: List of devices being used. By default, all `jax.local_devices`
            in the specified `backend` will be used.
        backend: Backend for computation. By default, `jax.default_backend` will
            be used.
        wrap_return: Wrapper for the return value.  If None (default), nothing
            will be applied so the return values have an extra leading axis that
            represents each local device.

    Returns:
        Wrapped function that runs on multiple devices.
    """
    if backend is None:
        backend = jax.default_backend()
    if devices is None:
        devices = jax.local_devices(backend=backend)
    if wrap_return is None:
        wrap_return = _id_fn
    if kwargtypes is None:
        kwargtypes = dict()

    argtypes = [resolve_argtype_str(ty) for ty in argtypes]
    kwargtypes = {k: resolve_argtype_str(ty) for k, ty in kwargtypes.items()}

    static_broadcasted_argnums = [i for i, ty in enumerate(argtypes) if ty.is_static()]
    if any(ty.is_static for ty in kwargtypes.values()):
        raise ValueError("`Static`-argument are only allowed as positional arguments.")

    pmapped_f = jax.pmap(
        f,
        axis_name,
        static_broadcasted_argnums=static_broadcasted_argnums,
        devices=devices,
        backend=backend,
        **kwargs,
    )

    def wrapped_f(*args, **kwargs):
        if len(args) != len(argtypes):
            raise ValueError(
                "The number of positional args passed to the target function "
                "does not match with the `argtypes` specified in `autopmap` "
                f"({len(args)} != {len(argtypes)})"
            )
        for k in kwargs.keys():
            if k not in kwargtypes:
                raise ValueError(
                    "`kwargtypes` specified to `autopmap` does not contain "
                    f"specification of kwarg with key={k}"
                )
        args = [ty.transform(x, devices) for x, ty in zip(args, argtypes)]
        kwargs = {k: kwargtypes[k].transform(x, devices) for k, x in kwargs.items()}
        ret = pmapped_f(*args, **kwargs)
        return wrap_return(ret)

    return wrapped_f


def unshard(tree: _ArrayTree) -> List[_ArrayTree]:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError("PyTree to be unsharded is empty.")
    if len(leaves[0].shape) < 1:
        raise ValueError("Tree to be unsharded do not have a leading axis.")
    size = leaves[0].shape[0]
    return [jax.tree_util.tree_map(lambda x: x[i], tree) for i in range(size)]
