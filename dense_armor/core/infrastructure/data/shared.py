import hashlib
import pickle
import sys
import weakref
from collections.abc import Mapping
from multiprocessing import resource_tracker, shared_memory
from typing import Any

import numpy as np


def _cleanup_shm(owner: bool, *names: str) -> None:
    if owner:
        for name in names:
            try:
                shared_memory.SharedMemory(name=name).unlink()
            except FileNotFoundError:
                pass


def _attach(name: str) -> shared_memory.SharedMemory:
    opts: dict[str, Any] = {"track": False}
    try:
        return shared_memory.SharedMemory(name=name, **opts)
    except TypeError:
        shm = shared_memory.SharedMemory(name=name)
        if sys.platform != "win32":
            resource_tracker.unregister(
                getattr(shm, "_name", shm.name), "shared_memory"
            )
        return shm


def _check_safe_pickle(item) -> None:
    if isinstance(item, (type(None), bool, int, float, str, bytes)):
        return
    if isinstance(item, (tuple, list)):
        for x in item:
            _check_safe_pickle(x)
        return
    raise TypeError(f"unsupported type: {type(item)}")


class SharedArray:
    """A numeric numpy array in one shared memory block.

    Source: `papers/issue_13246.md`, comment 1364587359 (the `numpy`
    workaround); `papers/ppwwyyxx_demystify_ram.txt`, section "Measure
    RAM Usage".

    Platform: Linux and Windows both work. `/dev/shm` exists only on
    Linux; on Windows the block disappears when the last handle closes
    and `unlink()` does nothing.

    Args:
        arr: A numeric array. An empty array raises `ValueError`; a
            `dtype=object` array raises `TypeError`.
    """

    def __init__(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            raise ValueError("empty array")
        if arr.dtype == object:
            raise TypeError("dtype=object is not supported in shared memory")
        self._shape = arr.shape
        self._dtype = arr.dtype
        self._shm = shared_memory.SharedMemory(create=True, size=int(arr.nbytes))
        self._arr: np.ndarray = np.ndarray(
            self._shape, dtype=self._dtype, buffer=self._shm.buf
        )
        self._arr[:] = arr
        self._arr.flags.writeable = False
        self._owner = True
        self._finalizer = weakref.finalize(
            self, _cleanup_shm, self._owner, self._shm.name
        )

    @property
    def array(self) -> np.ndarray:
        return self._arr

    @property
    def names(self) -> list[str]:
        return [self._shm.name]

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        if self._owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            self._owner = False

    def __len__(self) -> int:
        return self._shape[0] if self._shape else 1

    def __getstate__(self) -> dict:
        return {
            "name": self._shm.name,
            "shape": self._shape,
            "dtype": self._dtype,
            "owner": False,
        }

    def __setstate__(self, state: dict) -> None:
        self._shm = _attach(state["name"])
        self._shape, self._dtype = state["shape"], state["dtype"]
        self._arr = np.ndarray(self._shape, dtype=self._dtype, buffer=self._shm.buf)
        self._arr.flags.writeable = False
        self._owner = state["owner"]
        self._finalizer = weakref.finalize(
            self, _cleanup_shm, self._owner, self._shm.name
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unlink()
        self.close()


class SharedStrings:
    """A list of strings as one UTF-8 blob plus int64 offsets.

    Source: `papers/pr_191555.md` (the design); `papers/issue_13246.md`,
    comment 436632186 (a huge list grows memory, a numpy array does not).

    Platform: Linux and Windows both work. `/dev/shm` exists only on
    Linux; on Windows `unlink()` does nothing.

    Args:
        strings: A non-empty list of `str`. An empty list raises
            `ValueError`, and so does a list whose strings are all empty
            (a `SharedMemory` block cannot have size 0).
    """

    def __init__(self, strings: list[str]) -> None:
        if not strings:
            raise ValueError("empty list")
        self._n = len(strings)
        encoded = [s.encode("utf-8") for s in strings]
        self._offsets: np.ndarray = np.zeros(self._n + 1, dtype=np.int64)
        self._offsets[1:] = np.cumsum([len(e) for e in encoded])
        total_bytes = int(self._offsets[-1])
        if total_bytes == 0:
            raise ValueError("zero bytes")

        self._shm_data = shared_memory.SharedMemory(create=True, size=total_bytes)
        self._shm_offsets = shared_memory.SharedMemory(
            create=True, size=int(self._offsets.nbytes)
        )
        self._data: np.ndarray = np.ndarray(
            (total_bytes,), dtype=np.uint8, buffer=self._shm_data.buf
        )
        self._off_arr: np.ndarray = np.ndarray(
            (self._n + 1,), dtype=np.int64, buffer=self._shm_offsets.buf
        )
        self._off_arr[:] = self._offsets
        self._data[:] = np.frombuffer(b"".join(encoded), dtype=np.uint8)
        self._owner = True
        self._finalizer = weakref.finalize(
            self, _cleanup_shm, self._owner, self._shm_data.name, self._shm_offsets.name
        )

    @property
    def names(self) -> list[str]:
        return [self._shm_data.name, self._shm_offsets.name]

    def close(self) -> None:
        self._shm_data.close()
        self._shm_offsets.close()

    def unlink(self) -> None:
        if self._owner:
            for shm in (self._shm_data, self._shm_offsets):
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            self._owner = False

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self._n))]
        if i < 0:
            i += self._n
        if i < 0 or i >= self._n:
            raise IndexError
        start, end = self._off_arr[i], self._off_arr[i + 1]
        return bytes(self._data[start:end]).decode("utf-8")

    def __getstate__(self) -> dict:
        return {
            "name_data": self._shm_data.name,
            "name_offsets": self._shm_offsets.name,
            "n": self._n,
            "owner": False,
        }

    def __setstate__(self, state: dict) -> None:
        self._shm_data = _attach(state["name_data"])
        self._shm_offsets = _attach(state["name_offsets"])
        self._n = state["n"]
        self._data = np.ndarray(
            (self._shm_data.size,), dtype=np.uint8, buffer=self._shm_data.buf
        )
        self._off_arr = np.ndarray(
            (self._n + 1,), dtype=np.int64, buffer=self._shm_offsets.buf
        )
        self._owner = state["owner"]
        self._finalizer = weakref.finalize(
            self, _cleanup_shm, self._owner, self._shm_data.name, self._shm_offsets.name
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unlink()
        self.close()


class SharedList:
    """A flat buffer of pickled items plus int64 offsets.

    Source: `papers/pr_191555.md` (the design). The `multiprocessing`
    docs for protocol 5.

    What pickles and what does not: any object with a working
    `__getstate__` / `__setstate__` (custom classes, nested dicts,
    cyclic structures, numpy arrays, tensors) goes through. Objects
    that are not picklable (database connections, open file handles,
    sockets, threads, locks, lambdas, generators) raise `TypeError`
    from pickle. For a connection, put the *recipe* (the URL, the
    credentials, the path) in the container, and open the connection
    in the worker, one per process.

    Trust: reading an item calls `pickle.loads` on bytes in shared
    memory, so every local process that can attach the block is trusted
    to write only data it owns. `allow_pickle=False` restricts what is
    written at construction (`None`, `bool`, `int`, `float`, `str`,
    `bytes`, and `tuple` / `list` of these; anything else raises
    `TypeError`); it does not change how items are read.

    Platform: Linux and Windows both work. `/dev/shm` exists only on
    Linux; on Windows `unlink()` does nothing.

    Args:
        items: A non-empty list. An empty list raises `ValueError`.
        allow_pickle: If False, only the safe types above are accepted.
    """

    def __init__(self, items: list, allow_pickle: bool = True) -> None:
        if not items:
            raise ValueError("empty list")
        self._n, self._allow_pickle = len(items), allow_pickle
        encoded = []
        for item in items:
            if not allow_pickle:
                _check_safe_pickle(item)
            encoded.append(pickle.dumps(item, protocol=5))
        self._offsets: np.ndarray = np.zeros(self._n + 1, dtype=np.int64)
        self._offsets[1:] = np.cumsum([len(e) for e in encoded])
        total_bytes = int(self._offsets[-1])

        self._shm_data = shared_memory.SharedMemory(create=True, size=total_bytes)
        self._shm_offsets = shared_memory.SharedMemory(
            create=True, size=int(self._offsets.nbytes)
        )
        self._data: np.ndarray = np.ndarray(
            (total_bytes,), dtype=np.uint8, buffer=self._shm_data.buf
        )
        self._off_arr: np.ndarray = np.ndarray(
            (self._n + 1,), dtype=np.int64, buffer=self._shm_offsets.buf
        )
        self._off_arr[:] = self._offsets
        self._data[:] = np.frombuffer(b"".join(encoded), dtype=np.uint8)
        self._owner = True
        self._finalizer = weakref.finalize(
            self, _cleanup_shm, self._owner, self._shm_data.name, self._shm_offsets.name
        )

    @property
    def names(self) -> list[str]:
        return [self._shm_data.name, self._shm_offsets.name]

    def close(self) -> None:
        self._shm_data.close()
        self._shm_offsets.close()

    def unlink(self) -> None:
        if self._owner:
            for shm in (self._shm_data, self._shm_offsets):
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            self._owner = False

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self._n))]
        if i < 0:
            i += self._n
        if i < 0 or i >= self._n:
            raise IndexError
        start, end = self._off_arr[i], self._off_arr[i + 1]
        return pickle.loads(bytes(self._data[start:end]))

    def __getstate__(self) -> dict:
        return {
            "name_data": self._shm_data.name,
            "name_offsets": self._shm_offsets.name,
            "n": self._n,
            "owner": False,
            "allow_pickle": self._allow_pickle,
        }

    def __setstate__(self, state: dict) -> None:
        self._shm_data = _attach(state["name_data"])
        self._shm_offsets = _attach(state["name_offsets"])
        self._n, self._allow_pickle = state["n"], state["allow_pickle"]
        self._data = np.ndarray(
            (self._shm_data.size,), dtype=np.uint8, buffer=self._shm_data.buf
        )
        self._off_arr = np.ndarray(
            (self._n + 1,), dtype=np.int64, buffer=self._shm_offsets.buf
        )
        self._owner = state["owner"]
        self._finalizer = weakref.finalize(
            self, _cleanup_shm, self._owner, self._shm_data.name, self._shm_offsets.name
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unlink()
        self.close()


class SharedDict(Mapping):
    """Separate key and value buffers with offsets.

    Source: `papers/pr_191555.md` (the design). The `hash()` note is from
    `papers/ppwwyyxx_demystify_ram.txt`, section "Python objects".

    What pickles: see `SharedList`. Connections, file handles,
    sockets, threads, locks, lambdas and generators are not picklable:
    put the recipe in the container and open the resource in the worker.

    Declared limitation: keys are compared by the bytes of their
    pickle, not by Python equality. A Python `dict` treats `1`, `1.0`
    and `True` as the same key; this class treats them as distinct
    keys. The reason is the cost of a lookup: a digest-only match would
    unpickle every candidate key, which is slower than the byte
    comparison and defeats the sorted digest index. The page
    `docs/api/data_shared.md` repeats the limitation in the Choices
    section.

    Trust: reading a value calls `pickle.loads` on bytes in shared
    memory, so every local process that can attach the blocks is trusted
    to write only data it owns. `allow_pickle=False` restricts keys and
    values at construction to the safe types of `SharedList`.

    Platform: Linux and Windows both work. `/dev/shm` exists only on
    Linux; on Windows `unlink()` does nothing.

    Args:
        mapping: A non-empty dict. An empty dict raises `ValueError`.
        allow_pickle: If False, only the safe types are accepted.
    """

    def __init__(self, mapping: dict, allow_pickle: bool = True) -> None:
        if not mapping:
            raise ValueError("empty dict")
        self._n, self._allow_pickle = len(mapping), allow_pickle
        keys, vals = list(mapping.keys()), list(mapping.values())
        if not allow_pickle:
            for k in keys:
                _check_safe_pickle(k)
            for v in vals:
                _check_safe_pickle(v)
        enc_keys = [pickle.dumps(k, protocol=5) for k in keys]
        enc_vals = [pickle.dumps(v, protocol=5) for v in vals]

        self._key_offsets: np.ndarray = np.zeros(self._n + 1, dtype=np.int64)
        self._key_offsets[1:] = np.cumsum([len(e) for e in enc_keys])
        self._val_offsets: np.ndarray = np.zeros(self._n + 1, dtype=np.int64)
        self._val_offsets[1:] = np.cumsum([len(e) for e in enc_vals])

        total_key_bytes = int(self._key_offsets[-1])
        total_val_bytes = int(self._val_offsets[-1])

        self._shm_keys = shared_memory.SharedMemory(create=True, size=total_key_bytes)
        self._shm_key_offsets = shared_memory.SharedMemory(
            create=True, size=int(self._key_offsets.nbytes)
        )
        self._shm_vals = shared_memory.SharedMemory(create=True, size=total_val_bytes)
        self._shm_val_offsets = shared_memory.SharedMemory(
            create=True, size=int(self._val_offsets.nbytes)
        )

        self._keys_data: np.ndarray = np.ndarray(
            (total_key_bytes,), dtype=np.uint8, buffer=self._shm_keys.buf
        )
        self._key_off_arr: np.ndarray = np.ndarray(
            (self._n + 1,), dtype=np.int64, buffer=self._shm_key_offsets.buf
        )
        self._vals_data: np.ndarray = np.ndarray(
            (total_val_bytes,), dtype=np.uint8, buffer=self._shm_vals.buf
        )
        self._val_off_arr: np.ndarray = np.ndarray(
            (self._n + 1,), dtype=np.int64, buffer=self._shm_val_offsets.buf
        )

        self._key_off_arr[:] = self._key_offsets
        self._val_off_arr[:] = self._val_offsets
        self._keys_data[:] = np.frombuffer(b"".join(enc_keys), dtype=np.uint8)
        self._vals_data[:] = np.frombuffer(b"".join(enc_vals), dtype=np.uint8)

        digests = [
            int.from_bytes(hashlib.blake2b(k, digest_size=8).digest(), "little")
            for k in enc_keys
        ]
        self._digests = np.array(digests, dtype=np.uint64)
        self._sort_idx = np.argsort(self._digests)

        self._shm_digests = shared_memory.SharedMemory(
            create=True, size=int(self._digests.nbytes)
        )
        self._shm_sort_idx = shared_memory.SharedMemory(
            create=True, size=int(self._sort_idx.nbytes)
        )
        self._digest_arr: np.ndarray = np.ndarray(
            (self._n,), dtype=np.uint64, buffer=self._shm_digests.buf
        )
        self._sort_idx_arr: np.ndarray = np.ndarray(
            (self._n,), dtype=np.int64, buffer=self._shm_sort_idx.buf
        )

        self._digest_arr[:] = self._digests[self._sort_idx]
        self._sort_idx_arr[:] = self._sort_idx

        self._owner = True
        self._finalizer = weakref.finalize(
            self,
            _cleanup_shm,
            self._owner,
            self._shm_keys.name,
            self._shm_key_offsets.name,
            self._shm_vals.name,
            self._shm_val_offsets.name,
            self._shm_digests.name,
            self._shm_sort_idx.name,
        )

    @property
    def names(self) -> list[str]:
        return [
            self._shm_keys.name,
            self._shm_key_offsets.name,
            self._shm_vals.name,
            self._shm_val_offsets.name,
            self._shm_digests.name,
            self._shm_sort_idx.name,
        ]

    def close(self) -> None:
        for shm in (
            self._shm_keys,
            self._shm_key_offsets,
            self._shm_vals,
            self._shm_val_offsets,
            self._shm_digests,
            self._shm_sort_idx,
        ):
            shm.close()

    def unlink(self) -> None:
        if self._owner:
            for shm in (
                self._shm_keys,
                self._shm_key_offsets,
                self._shm_vals,
                self._shm_val_offsets,
                self._shm_digests,
                self._shm_sort_idx,
            ):
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            self._owner = False

    def __len__(self) -> int:
        return self._n

    def _get_val(self, idx: int):
        start, end = self._val_off_arr[idx], self._val_off_arr[idx + 1]
        return pickle.loads(bytes(self._vals_data[start:end]))

    def _get_key_bytes(self, idx: int) -> bytes:
        start, end = self._key_off_arr[idx], self._key_off_arr[idx + 1]
        return bytes(self._keys_data[start:end])

    def __getitem__(self, key):
        kb = pickle.dumps(key, protocol=5)
        d = int.from_bytes(hashlib.blake2b(kb, digest_size=8).digest(), "little")
        pos = np.searchsorted(self._digest_arr, d)
        if pos < len(self._digest_arr) and self._digest_arr[pos] == d:
            while pos < len(self._digest_arr) and self._digest_arr[pos] == d:
                real_idx = self._sort_idx_arr[pos]
                if self._get_key_bytes(real_idx) == kb:
                    return self._get_val(real_idx)
                pos += 1
        raise KeyError(key)

    def __iter__(self):
        for i in range(self._n):
            yield pickle.loads(self._get_key_bytes(i))

    def __getstate__(self) -> dict:
        return {
            "n": self._n,
            "allow_pickle": self._allow_pickle,
            "names": self.names,
            "owner": False,
        }

    def __setstate__(self, state: dict) -> None:
        self._n, self._allow_pickle = state["n"], state["allow_pickle"]
        self._shm_keys = _attach(state["names"][0])
        self._shm_key_offsets = _attach(state["names"][1])
        self._shm_vals = _attach(state["names"][2])
        self._shm_val_offsets = _attach(state["names"][3])
        self._shm_digests = _attach(state["names"][4])
        self._shm_sort_idx = _attach(state["names"][5])
        self._keys_data = np.ndarray(
            (self._shm_keys.size,), dtype=np.uint8, buffer=self._shm_keys.buf
        )
        self._key_off_arr = np.ndarray(
            (self._n + 1,), dtype=np.int64, buffer=self._shm_key_offsets.buf
        )
        self._vals_data = np.ndarray(
            (self._shm_vals.size,), dtype=np.uint8, buffer=self._shm_vals.buf
        )
        self._val_off_arr = np.ndarray(
            (self._n + 1,), dtype=np.int64, buffer=self._shm_val_offsets.buf
        )
        self._digest_arr = np.ndarray(
            (self._n,), dtype=np.uint64, buffer=self._shm_digests.buf
        )
        self._sort_idx_arr = np.ndarray(
            (self._n,), dtype=np.int64, buffer=self._shm_sort_idx.buf
        )
        self._owner = state["owner"]
        self._finalizer = weakref.finalize(self, _cleanup_shm, self._owner, *self.names)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unlink()
        self.close()


def to_shared(obj, allow_pickle: bool = True):
    """Convert a Python object to a shared container.

    Args:
        obj: An `ndarray` (`SharedArray`), a list or tuple of `str`
            (`SharedStrings`), any other list or tuple (`SharedList`) or
            a dict (`SharedDict`). Anything else raises `TypeError`; an
            empty list or tuple raises `ValueError`.
        allow_pickle: Passed to `SharedList` and `SharedDict`; ignored for
            arrays and lists of `str`.

    Returns:
        The shared container.
    """
    if isinstance(obj, np.ndarray):
        return SharedArray(obj)
    if isinstance(obj, dict):
        return SharedDict(obj, allow_pickle=allow_pickle)
    if isinstance(obj, (list, tuple)):
        lst = list(obj)
        if not lst:
            raise ValueError("empty list")
        if all(isinstance(x, str) for x in lst):
            return SharedStrings(lst)
        return SharedList(lst, allow_pickle=allow_pickle)
    raise TypeError(f"Cannot convert {type(obj)} to a shared container")
