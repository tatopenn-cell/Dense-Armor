import hashlib
import pickle
import weakref
from collections.abc import Mapping
from multiprocessing import shared_memory

import numpy as np


class SharedArray:
    """A numeric numpy array in one shared memory block.

    Source: `papers/issue_13246.md`, comment 1364587359 (the `numpy`
    workaround) and the Editor note; `papers/ppwwyyxx_demystify_ram.txt`,
    section "Measure RAM Usage".
    after `fork`; `papers/ppwwyyxx_demystify_ram.txt`, section "Measure
    RAM Usage" for the difference between RSS, PSS and USS.
    The array is read-only (`flags.writeable` is False) so that a read
    does not change the refcount.
    """

    def __init__(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            raise ValueError("empty array")
        self._shape = arr.shape
        self._dtype = arr.dtype
        self._shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        self._arr: np.ndarray = np.ndarray(
            self._shape, dtype=self._dtype, buffer=self._shm.buf
        )
        self._arr[:] = arr
        self._arr.flags.writeable = False
        self._owner = True
        self._finalizer = weakref.finalize(
            self, self._cleanup, self._owner, self._shm.name
        )

    def _cleanup(self, owner: bool, name: str) -> None:
        if owner:
            try:
                shared_memory.SharedMemory(name=name).unlink()
            except FileNotFoundError:
                pass

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
        self._shm = shared_memory.SharedMemory(name=state["name"])
        self._shape, self._dtype = state["shape"], state["dtype"]
        self._arr = np.ndarray(self._shape, dtype=self._dtype, buffer=self._shm.buf)
        self._arr.flags.writeable = False
        self._owner = state["owner"]
        self._finalizer = weakref.finalize(
            self, self._cleanup, self._owner, self._shm.name
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unlink()
        self.close()


class SharedStrings:
    """A list of strings as one UTF-8 blob plus int64 offsets.

    Source: `papers/pr_191555.md` (the design, not the numbers);
    `papers/issue_13246.md`, comment 436632186 (the case of the list of paths).
    The motivation is the case of the issue (`papers/issue_13246.md`): a
    list of file paths read by index. `__getitem__(i)` decodes only item `i`,
    so it does not touch the pages of the other items.
    """

    def __init__(self, strings: list[str]) -> None:
        if not strings:
            raise ValueError("empty list")
        self._n = len(strings)
        encoded = [s.encode("utf-8") for s in strings]
        self._offsets = np.zeros(self._n + 1, dtype=np.int64)
        self._offsets[1:] = np.cumsum([len(e) for e in encoded])
        total_bytes = self._offsets[-1]
        if total_bytes == 0:
            raise ValueError("zero bytes")

        self._shm_data = shared_memory.SharedMemory(create=True, size=total_bytes)
        self._shm_offsets = shared_memory.SharedMemory(
            create=True, size=self._offsets.nbytes
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
            self,
            self._cleanup,
            self._owner,
            self._shm_data.name,
            self._shm_offsets.name,
        )

    def _cleanup(self, owner: bool, name_data: str, name_off: str) -> None:
        if owner:
            for name in (name_data, name_off):
                try:
                    shared_memory.SharedMemory(name=name).unlink()
                except FileNotFoundError:
                    pass

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
        self._shm_data = shared_memory.SharedMemory(name=state["name_data"])
        self._shm_offsets = shared_memory.SharedMemory(name=state["name_offsets"])
        self._n = state["n"]
        self._data = np.ndarray(
            (self._shm_data.size,), dtype=np.uint8, buffer=self._shm_data.buf
        )
        self._off_arr = np.ndarray(
            (self._n + 1,), dtype=np.int64, buffer=self._shm_offsets.buf
        )
        self._owner = state["owner"]
        self._finalizer = weakref.finalize(
            self,
            self._cleanup,
            self._owner,
            self._shm_data.name,
            self._shm_offsets.name,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unlink()
        self.close()


class SharedList:
    """A flat buffer of pickled items plus int64 offsets.

    Source: `papers/pr_191555.md` (the design, not the numbers); the
    pickle protocol 5 is from the `multiprocessing` docs, section
    "SharedMemory".
    By default (`allow_pickle=True`), the items are pickled with protocol 5.
    Warning: pickle from shared memory trusts every local process, as
    noted in the security section of the `multiprocessing` docs.
    """

    def __init__(self, items: list, allow_pickle: bool = True) -> None:
        if not items:
            raise ValueError("empty list")
        self._n, self._allow_pickle = len(items), allow_pickle
        encoded = []
        for item in items:
            if not allow_pickle and not isinstance(
                item, (type(None), bool, int, float, str, bytes, tuple, list)
            ):
                raise TypeError("unsupported type")
            encoded.append(pickle.dumps(item, protocol=5))
        self._offsets = np.zeros(self._n + 1, dtype=np.int64)
        self._offsets[1:] = np.cumsum([len(e) for e in encoded])
        total_bytes = self._offsets[-1]
        if total_bytes == 0:
            raise ValueError("zero bytes")

        self._shm_data = shared_memory.SharedMemory(create=True, size=total_bytes)
        self._shm_offsets = shared_memory.SharedMemory(
            create=True, size=self._offsets.nbytes
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
            self,
            self._cleanup,
            self._owner,
            self._shm_data.name,
            self._shm_offsets.name,
        )

    def _cleanup(self, owner: bool, name_data: str, name_off: str) -> None:
        if owner:
            for name in (name_data, name_off):
                try:
                    shared_memory.SharedMemory(name=name).unlink()
                except FileNotFoundError:
                    pass

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
        self._shm_data = shared_memory.SharedMemory(name=state["name_data"])
        self._shm_offsets = shared_memory.SharedMemory(name=state["name_offsets"])
        self._n, self._allow_pickle = state["n"], state["allow_pickle"]
        self._data = np.ndarray(
            (self._shm_data.size,), dtype=np.uint8, buffer=self._shm_data.buf
        )
        self._off_arr = np.ndarray(
            (self._n + 1,), dtype=np.int64, buffer=self._shm_offsets.buf
        )
        self._owner = state["owner"]
        self._finalizer = weakref.finalize(
            self,
            self._cleanup,
            self._owner,
            self._shm_data.name,
            self._shm_offsets.name,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unlink()
        self.close()


class SharedDict(Mapping):
    """Separate key and value buffers with offsets.

    Source: `papers/pr_191555.md` (the design, not the numbers); the
    `hash()` note is from `papers/ppwwyyxx_demystify_ram.txt`, section
    "Python objects".
    The lookup uses `hashlib.blake2b(pickle_of_key, digest_size=8)` because
    `hash()` of a `str` differs between processes, as noted in
    `papers/ppwwyyxx_demystify_ram.txt` (the section on Python objects).
    A missing key raises `KeyError`.
    """

    def __init__(self, mapping: dict, allow_pickle: bool = True) -> None:
        if not mapping:
            raise ValueError("empty dict")
        self._n, self._allow_pickle = len(mapping), allow_pickle
        keys, vals = list(mapping.keys()), list(mapping.values())
        enc_keys = [pickle.dumps(k, protocol=5) for k in keys]
        enc_vals = [pickle.dumps(v, protocol=5) for v in vals]

        self._key_offsets = np.zeros(self._n + 1, dtype=np.int64)
        self._key_offsets[1:] = np.cumsum([len(e) for e in enc_keys])
        self._val_offsets = np.zeros(self._n + 1, dtype=np.int64)
        self._val_offsets[1:] = np.cumsum([len(e) for e in enc_vals])

        total_key_bytes, total_val_bytes = self._key_offsets[-1], self._val_offsets[-1]
        if total_key_bytes == 0 or total_val_bytes == 0:
            raise ValueError("zero bytes")

        self._shm_keys = shared_memory.SharedMemory(create=True, size=total_key_bytes)
        self._shm_key_offsets = shared_memory.SharedMemory(
            create=True, size=self._key_offsets.nbytes
        )
        self._shm_vals = shared_memory.SharedMemory(create=True, size=total_val_bytes)
        self._shm_val_offsets = shared_memory.SharedMemory(
            create=True, size=self._val_offsets.nbytes
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
            create=True, size=self._digests.nbytes
        )
        self._shm_sort_idx = shared_memory.SharedMemory(
            create=True, size=self._sort_idx.nbytes
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
            self._cleanup,
            self._owner,
            self._shm_keys.name,
            self._shm_key_offsets.name,
            self._shm_vals.name,
            self._shm_val_offsets.name,
            self._shm_digests.name,
            self._shm_sort_idx.name,
        )

    def _cleanup(self, owner: bool, *names: str) -> None:
        if owner:
            for name in names:
                try:
                    shared_memory.SharedMemory(name=name).unlink()
                except FileNotFoundError:
                    pass

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
        self._shm_keys = shared_memory.SharedMemory(name=state["names"][0])
        self._shm_key_offsets = shared_memory.SharedMemory(name=state["names"][1])
        self._shm_vals = shared_memory.SharedMemory(name=state["names"][2])
        self._shm_val_offsets = shared_memory.SharedMemory(name=state["names"][3])
        self._shm_digests = shared_memory.SharedMemory(name=state["names"][4])
        self._shm_sort_idx = shared_memory.SharedMemory(name=state["names"][5])
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
        self._finalizer = weakref.finalize(
            self, self._cleanup, self._owner, *self.names
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unlink()
        self.close()


def to_shared(obj, allow_pickle: bool = True):
    """Convert a Python object to a shared container."""
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
