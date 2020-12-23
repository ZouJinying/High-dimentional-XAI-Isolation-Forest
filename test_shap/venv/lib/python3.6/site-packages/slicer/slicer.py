""" Prototype. This is currently under active development.

TODO:
? Sparse
- Package integration (SHAP, EBM)
? Tensorflow
"""

from typing import Any, AnyStr, Union, List, Tuple, Optional, Dict
from abc import abstractmethod

import numbers

_available_modules = set()


class AtomicSlicer:
    """ Wrapping object that will unify slicing across data structures.

    What we support:
        Basic indexing (return references):
        - (start:stop:step) slicing
        - support ellipses
        Advanced indexing (return references):
        - integer array indexing
        - boolean array indexing? (discussion for later)

    Numpy Reference:
        Basic indexing (return views):
        - (start:stop:step) slicing
        - support ellipses and newaxis (alias for None)
        Advanced indexing (return copy):
        - integer array indexing, i.e. X[[1,2], [3,4]]
        - boolean array indexing
        - mixed array indexing (has integer array, ellipses, newaxis in same slice)
    """

    def __init__(
            self,
            o: Any,
            max_dim: Union[None, int, AnyStr] = "auto",
    ):
        """ Provides a consistent slicing API to the object provided.

        Args:
            o: Object to enable consistent slicing.
                Currently supports numpy dense arrays, recursive lists ending with list or numpy.
            max_dim: Max number of dimensions the wrapped object has.
                If set to "auto", max dimensions will be inferred. This comes at compute cost.
        """
        self.o = o
        self.max_dim = max_dim
        if self.max_dim == "auto":
            self.max_dim = UnifiedDataHandler.max_dim(o)

    def __repr__(self) -> AnyStr:
        """ Override default repr for human readability.

        Returns:
            String to display.
        """
        return f"{self.__class__.__name__}({self.o.__repr__()})"

    def __getitem__(self, item: Any) -> Any:
        """ Consistent slicing into wrapped object.

        Args:
            item: Slicing key of type integer or slice.

        Returns:
            Sliced object.

        Raises:
            ValueError: If slicing is not compatible with wrapped object.
        """
        # Turn item into tuple if not already.
        index_tup = _unify_slice(item, self.max_dim)

        # Slice according to object type.
        return UnifiedDataHandler.slice(self.o, index_tup, self.max_dim)


# TODO: Document.
class Slicer:
    def __init__(self, *args, **kwargs):
        # NOTE: Protected attributes.
        self._max_dim = 0

        # NOTE: Private attributes.
        self._anon = []
        self._objects = {}
        self._aliases = {}
        self._alias_lookup = None

        # Go through unnamed objects / aliases
        self.__setattr__("o", args)

        # Go through named objects / aliases
        for key, value in kwargs.items():
            self.__setattr__(key, value)

        # Generate default aliases only if one object and no aliases exist
        objects_len = len(self._objects)
        anon_len = len(self._anon)
        aliases_len = len(self._aliases)
        if ((objects_len == 1) ^ (anon_len == 1)) and aliases_len == 0:
            obj = None
            for _, t in self._iter_tracked():
                obj = t

            generated_aliases = UnifiedDataHandler.default_alias(obj.o)
            for generated_alias in generated_aliases:
                self.__setattr__(generated_alias._name, generated_alias)

    def _update_max_dim(self, tracked):
        self._max_dim = max(self._max_dim, max(tracked.dim, default=-1) + 1)

    def _iter_tracked(self, include_aliases=False):
        for tracked in self._anon:
            yield "o", tracked
        for name, tracked in self._objects.items():
            yield name, tracked
        if include_aliases:
            for name, tracked in self._aliases.items():
                yield name, tracked

    def _recompute_max_dim(self):
        self._max_dim = max(
            [
                max(o.dim, default=-1) + 1
                for _, o in self._iter_tracked()
            ],
            default=0
        )

    def _reduced_o(self, tracked):
        os = [t.o for t in tracked]
        os = os[0] if len(os) == 1 else os
        return os

    def __setattr__(self, key, value):
        """ Override default setattr to sync tracking of slicer.

        Args:
            key: Name of tracked attribute.
            value: Either an Obj, Alias or Python Object.
        """
        if key.startswith("_"):
            return super(Slicer, self).__setattr__(key, value)

        if isinstance(value, Alias):
            value._name = key
            self._aliases[key] = value

            # Build lookup (for perf)
            if self._alias_lookup is None:
                self._alias_lookup = AliasLookup(self._aliases)
            else:
                self._alias_lookup.update(value)
        else:
            if key == "o":
                tracked = [Obj(x) if not isinstance(x, Obj) else x for x in value]
                self._anon = tracked
                for t in tracked:
                    self._update_max_dim(t)

                os = self._reduced_o(self._anon)
                super(Slicer, self).__setattr__(key, os)
            else:
                value = Obj(value) if not isinstance(value, Obj) else value
                value._name = key
                self._objects[key] = value
                self._update_max_dim(value)
                super(Slicer, self).__setattr__(key, value.o)

    def __getattr__(self, item) -> Any:
        """ Override default getattr to return tracked attribute.

        Args:
            item: Name of tracked attribute.
        Returns:
            Corresponding object.
        """
        if item.startswith("_"):
            return super(Slicer, self).__getattr__(item)

        if item == "o":
            return self._reduced_o(self._anon)
        else:
            tracked = self._objects.get(item, None)
            if tracked is None:
                tracked = self._aliases.get(item, None)

            if tracked is None:
                raise AttributeError(f"Attribute '{item}' does not exist.")

            return tracked.o

    def __delattr__(self, item):
        """ Override default delattr to remove tracked attribute.

        Args:
            item: Name of tracked attribute to delete.
        """
        if item.startswith("_"):
            return super(Slicer, self).__delattr__(item)

        # Sync private attributes that help track
        self._objects.pop(item, None)
        self._aliases.pop(item, None)
        if item == "o":
            self._anon.clear()

        # Recompute max_dim
        self._recompute_max_dim()

        # Recompute alias lookup
        # NOTE: This doesn't use diff-style deletes, but we don't care (not a perf target).
        self._alias_lookup = AliasLookup(self._aliases)

        # TODO: Mutate and check interactively what it does
        super(Slicer, self).__delattr__(item)

    def __repr__(self) -> AnyStr:
        """ Override default repr for human readability.

        Returns:
            String to display.
        """
        orig = self.__dict__
        di = {}
        for key, value in orig.items():
            if not key.startswith("_"):
                di[key] = value
        return f"{self.__class__.__name__}({str(di)})"

    def __getitem__(self, item):
        index_tup = _unify_slice(item, self._max_dim, self._alias_lookup)
        new_args = []
        new_kwargs = {}
        for name, tracked in self._iter_tracked(include_aliases=True):
            if len(tracked.dim) == 0:  # No slice on empty dim
                new_tracked = tracked
            else:
                index_slicer = AtomicSlicer(index_tup, max_dim=1)
                slicer_index = index_slicer[tracked.dim]
                sliced_o = tracked[slicer_index]
                sliced_dim = _resolve_dim(index_tup, tracked.dim)

                new_tracked = tracked.__class__(sliced_o, sliced_dim)
                new_tracked._name = tracked._name

            if name == "o":
                new_args.append(new_tracked)
            else:
                new_kwargs[name] = new_tracked
        return self.__class__(*new_args, **new_kwargs)


def _resolve_dim(slicer_index: Tuple, slicer_dim: List) -> List:
    """ Extracts new dim after applying slicing index and maps it back to the original index list. """

    new_slicer_dim = []
    reduced_mask = []

    for _, curr_idx in enumerate(slicer_index):
        if isinstance(curr_idx, (tuple, list, slice)):
            reduced_mask.append(0)
        else:
            reduced_mask.append(1)

    for curr_dim in slicer_dim:
        if reduced_mask[curr_dim] == 0:
            new_slicer_dim.append(curr_dim - sum(reduced_mask[:curr_dim]))

    return new_slicer_dim


def _unify_slice(item: Any, max_dim: int, alias_lookup=None) -> Tuple:
    """ Resolves aliases and ellipses in a slice item.

    Args:
        item: Slicing key that is passed to __getitem__.
        max_dim: Max dimension of object to be sliced.
        alias_lookup: AliasLookup structure.

    Returns:
        A tuple representation of the item.
    """
    item = _normalize_slice_key(item)
    index_tup = _normalize_subkey_types(item)
    index_tup = _handle_newaxis_ellipses(index_tup, max_dim)
    if alias_lookup:
        index_tup = _handle_aliases(index_tup, alias_lookup)
    return index_tup


def _normalize_subkey_types(index_tup: tuple) -> Tuple:
    """ Casts subkeys into basic types such as int.

    Args:
        key: Slicing key that is passed within __getitem__.

    Returns:
        Tuple with subkeys casted to basic types.
    """
    new_index_tup = []  # Gets casted to tuple at the end

    # TODO: Profile safe instance checks and type sets.
    np_int_types = {'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'}
    for subkey in index_tup:
        if _safe_isinstance(subkey, "numpy", np_int_types):
            new_subkey = int(subkey)
        elif _safe_isinstance(subkey, "numpy", "ndarray"):
            if len(subkey.shape) == 1:
                new_subkey = subkey.tolist()
            else:
                raise ValueError(f"Cannot use array of shape {subkey.shape} as subkey.")
        else:
            new_subkey = subkey

        new_index_tup.append(new_subkey)
    return tuple(new_index_tup)


def _normalize_slice_key(key: Any) -> Tuple:
    """ Normalizes slice key into always being a top-level tuple.

    Args:
        key: Slicing key that is passed within __getitem__.

    Returns:
        Expanded slice as a tuple.
    """
    if not isinstance(key, tuple):
        return (key,)
    else:
        return key


def _handle_newaxis_ellipses(index_tup: tuple, max_dim: int) -> Tuple:
    """ Expands newaxis and ellipses within a slice for simplification.
    This code is mostly adapted from: https://github.com/clbarnes/h5py_like/blob/master/h5py_like/shape_utils.py#L111

    Args:
        index_tup: Slicing key as a tuple.
        max_dim: Maximum number of dimensions in the respective sliceable object.

    Returns:
        Expanded slice as a tuple.
    """
    non_indexes = (None, Ellipsis)
    concrete_indices = sum(idx not in non_indexes for idx in index_tup)
    index_list = []
    # newaxis_at = []
    has_ellipsis = False
    int_count = 0
    for item in index_tup:
        if isinstance(item, numbers.Number):
            int_count += 1

        # NOTE: If we need locations of new axis, re-enable this.
        if item is None:  # pragma: no cover
            pass
            # newaxis_at.append(len(index_list) + len(newaxis_at) - int_count)
        elif item == Ellipsis:
            if has_ellipsis:  # pragma: no cover
                raise IndexError("an index can only have a single ellipsis ('...')")
            has_ellipsis = True
            initial_len = len(index_list)
            while len(index_list) + (concrete_indices - initial_len) < max_dim:
                index_list.append(slice(None))
        else:
            index_list.append(item)

    if len(index_list) > max_dim:  # pragma: no cover
        raise IndexError("too many indices for array")
    while len(index_list) < max_dim:
        index_list.append(slice(None))

    # return index_list, newaxis_at
    return tuple(index_list)


def _handle_aliases(index_tup: Tuple, alias_lookup) -> Tuple:
    new_index_tup = []

    def resolve(item, dim):
        if isinstance(item, slice):
            return item
        # Replace element if in alias lookup, otherwise use original.
        item = alias_lookup.get(dim, item, item)
        return item

    # Go through each element within the index and resolve if needed.
    for dim, item in enumerate(index_tup):
        if isinstance(item, list):
            new_item = []
            for sub_item in item:
                new_item.append(resolve(sub_item, dim))
        else:
            new_item = resolve(item, dim)
        new_index_tup.append(new_item)

    return tuple(new_index_tup)


class Tracked(AtomicSlicer):
    def __init__(self, o: Any, dim: Union[int, List, None] = None):
        super().__init__(o)

        # Protected attribute that can be overriden.
        self._name = None

        # Place dim into coordinate form.
        if dim is None:
            self.dim = list(range(self.max_dim))
        elif isinstance(dim, int):
            self.dim = [dim]
        elif isinstance(dim, list):
            self.dim = dim
        else:  # pragma: no cover
            raise ValueError(f"Cannot handle dim of type: {type(dim)}")


# TODO: Document.
class Obj(Tracked):
    def __init__(self, o, dim=None):
        super().__init__(o, dim)


# NOTE: Alias is different at init as dim is now required.
class Alias(Tracked):
    def __init__(self, o, dim):
        if not (isinstance(dim, int) or isinstance(dim, list) and len(dim) <= 1):  # pragma: no cover
            raise ValueError("Aliases must track a single dimension")
        super().__init__(o, dim)


# TODO: Fix the lookup
class AliasLookup:
    def __init__(self, aliases):
        self._lookup = {}

        # Populate lookup and merge indexes.
        for _, alias in aliases.items():
            self.update(alias)

    def update(self, alias):
        if alias.dim is None or len(alias.dim) == 0:
            return

        dim = alias.dim[0]
        if dim not in self._lookup:
            self._lookup[dim] = {}

        dim_lookup = self._lookup[dim]
        # NOTE: Alias must be backed by either a list or dictionary.
        itr = enumerate(alias.o) if isinstance(alias.o, list) else alias.o.items()
        for i, x in itr:
            if x not in dim_lookup:
                dim_lookup[x] = set()
            dim_lookup[x].add(i)

    def get(self, dim, target, default=None):
        if dim not in self._lookup:
            return default

        indexes = self._lookup[dim].get(target, None)
        if indexes is None:
            return default

        if len(indexes) == 1:
            return next(iter(indexes))
        else:
            return list(indexes)


class BaseHandler:
    @classmethod
    @abstractmethod
    def head_slice(cls, o, index_tup, max_dim):
        raise NotImplementedError()  # pragma: no cover

    @classmethod
    @abstractmethod
    def tail_slice(cls, o, tail_index, max_dim, flatten=True):
        raise NotImplementedError()  # pragma: no cover

    @classmethod
    @abstractmethod
    def max_dim(cls, o):
        raise NotImplementedError()  # pragma: no cover

    @classmethod
    def default_alias(cls, o):
        return []


class SeriesHandler(BaseHandler):
    @classmethod
    def head_slice(cls, o, index_tup, max_dim):
        head_index = index_tup[0]
        is_element = True if isinstance(head_index, int) else False
        sliced_o = o.iloc[head_index]

        return is_element, sliced_o, 1

    @classmethod
    def tail_slice(cls, o, tail_index, max_dim, flatten=True):
        # NOTE: Series only has one dimension,
        #       call slicer again to end the recursion.
        return AtomicSlicer(o, max_dim=max_dim)[tail_index]

    @classmethod
    def max_dim(cls, o):
        return len(o.shape)

    @classmethod
    def default_alias(cls, o):
        index_alias = Alias(o.index.to_list(), 0)
        index_alias._name = "index"
        return [index_alias]


class DataFrameHandler(BaseHandler):
    @classmethod
    def head_slice(cls, o, index_tup, max_dim):
        # NOTE: At head slice, we know there are two fixed dimensions.
        cut_index = index_tup
        is_element = True if isinstance(cut_index[-1], int) else False
        sliced_o = o.iloc[cut_index]

        return is_element, sliced_o, 2

    @classmethod
    def tail_slice(cls, o, tail_index, max_dim, flatten=True):
        # NOTE: Dataframe has fixed dimensions,
        #       call slicer again to end the recursion.
        return AtomicSlicer(o, max_dim=max_dim)[tail_index]

    @classmethod
    def max_dim(cls, o):
        return len(o.shape)

    @classmethod
    def default_alias(cls, o):
        index_alias = Alias(o.index.to_list(), 0)
        index_alias._name = "index"
        column_alias = Alias(o.columns.to_list(), 1)
        column_alias._name = "columns"
        return [index_alias, column_alias]


class ArrayHandler(BaseHandler):
    @classmethod
    def head_slice(cls, o, index_tup, max_dim):
        # Check if head is string
        head_index, tail_index = index_tup[0], index_tup[1:]
        cut = 1

        for sub_index in tail_index:
            cut += 1
            if isinstance(sub_index, str):
                break

        # Process native array dimensions
        cut_index = index_tup[:cut]
        is_element = True if isinstance(cut_index[-1], int) else False
        sliced_o = o[cut_index]

        return is_element, sliced_o, cut

    @classmethod
    def tail_slice(cls, o, tail_index, max_dim, flatten=True):
        if flatten:
            return AtomicSlicer(o, max_dim=max_dim)[tail_index]
        else:
            inner = [AtomicSlicer(e, max_dim=max_dim)[tail_index] for e in o]
            if _safe_isinstance(o, "numpy", "ndarray"):
                import numpy

                return numpy.array(inner)
            elif _safe_isinstance(o, "torch", "Tensor"):
                import torch

                if len(inner) > 0 and isinstance(inner[0], torch.Tensor):
                    return torch.stack(inner)
                else:
                    return torch.tensor(inner)
            else:
                raise ValueError(f"Cannot handle type {type(o)}.")  # pragma: no cover

    @classmethod
    def max_dim(cls, o):
        return len(o.shape)


class DictHandler(BaseHandler):
    @classmethod
    def head_slice(cls, o, index_tup, max_dim):
        head_index = index_tup[0]
        if isinstance(head_index, (tuple, list)) and len(index_tup) == 0:
            return False, o, 1

        if isinstance(head_index, (list, tuple)):
            return (
                False,
                {
                    sub_index: AtomicSlicer(o, max_dim=max_dim)[sub_index]
                    for sub_index in head_index
                },
                1,
            )
        elif isinstance(head_index, slice):
            if head_index == slice(None, None, None):
                return False, o, 1
            return False, o[head_index], 1
        else:
            return True, o[head_index], 1

    @classmethod
    def tail_slice(cls, o, tail_index, max_dim, flatten=True):
        if flatten:
            return AtomicSlicer(o, max_dim=max_dim)[tail_index]
        else:
            return {k: AtomicSlicer(e, max_dim=max_dim)[tail_index] for k, e in o.items()}

    @classmethod
    def max_dim(cls, o):
        return max([UnifiedDataHandler.max_dim(x) for x in o.values()], default=-1) + 1


class ListTupleHandler(BaseHandler):
    @classmethod
    def head_slice(cls, o, index_tup, max_dim):
        head_index = index_tup[0]
        if isinstance(head_index, (tuple, list)) and len(index_tup) == 0:
            return False, o, 1

        if isinstance(head_index, (list, tuple)):
            if len(head_index) == 0:
                return False, o, 1
            else:
                results = [
                    AtomicSlicer(o, max_dim=max_dim)[sub_index] for sub_index in head_index
                ]
                results = tuple(results) if isinstance(o, tuple) else results
                return False, results, 1
        elif isinstance(head_index, slice):
            return False, o[head_index], 1
        elif isinstance(head_index, int):
            return True, o[head_index], 1
        else:  # pragma: no cover
            raise ValueError(f"Invalid key {head_index} for {o}")

    @classmethod
    def tail_slice(cls, o, tail_index, max_dim, flatten=True):
        if flatten:
            return AtomicSlicer(o, max_dim=max_dim)[tail_index]
        else:
            results = [AtomicSlicer(e, max_dim=max_dim)[tail_index] for e in o]
            return tuple(results) if isinstance(o, tuple) else results

    @classmethod
    def max_dim(cls, o):
        return max([UnifiedDataHandler.max_dim(x) for x in o], default=-1) + 1


class UnifiedDataHandler:
    """ Registry that maps types to their unified slice calls."""

    """ Class attribute that maps type to their unified slice calls."""
    type_map = {
        ("builtins", "list"): ListTupleHandler,
        ("builtins", "tuple"): ListTupleHandler,
        ("builtins", "dict"): DictHandler,
        ("torch", "Tensor"): ArrayHandler,
        ("numpy", "ndarray"): ArrayHandler,
        ("pandas.core.frame", "DataFrame"): DataFrameHandler,
        ("pandas.core.series", "Series"): SeriesHandler,
    }

    @classmethod
    def slice(cls, o, index_tup, max_dim):
        # NOTE: Unified handles base cases such as empty tuples, which
        #       specialized handlers do not.
        if isinstance(index_tup, (tuple, list)) and len(index_tup) == 0:
            return o

        # Slice as delegated by data handler.
        o_type = _type_name(o)
        head_slice = cls.type_map[o_type].head_slice
        tail_slice = cls.type_map[o_type].tail_slice

        is_element, sliced_o, cut = head_slice(o, index_tup, max_dim)
        out = tail_slice(sliced_o, index_tup[cut:], max_dim - cut, is_element)
        return out

    @classmethod
    def max_dim(cls, o):
        o_type = _type_name(o)
        if o_type not in cls.type_map:
            return 0
        return cls.type_map[o_type].max_dim(o)

    @classmethod
    def default_alias(cls, o):
        o_type = _type_name(o)
        if o_type not in cls.type_map:
            return {}
        return cls.type_map[o_type].default_alias(o)


def _type_name(o: object) -> Tuple[str, str]:
    return o.__class__.__module__, o.__class__.__name__


def _safe_isinstance(o: object, module_name: str, type_name: Union[str, set, tuple]) -> bool:
    o_module, o_type = _type_name(o)
    if isinstance(type_name, str):
        return o_module == module_name and o_type == type_name
    else:
        return o_module == module_name and o_type in type_name

