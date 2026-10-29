import gc
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from dense_armor.utility.data import (
    SharedArray,
    SharedDict,
    SharedList,
    SharedStrings,
    to_shared,
)


def _read_index(args):
    handle, i = args
    return handle[i]


def test_shared_array_roundtrip():
    arr = np.array([1.0, 2.0, 3.0])
    with SharedArray(arr) as sa:
        assert np.array_equal(sa.array, arr)
        assert not sa.array.flags.writeable
        assert len(sa) == 3
        assert len(sa.names) == 1


def test_shared_array_empty():
    with pytest.raises(ValueError):
        SharedArray(np.array([]))


def test_shared_array_write_raises():
    with SharedArray(np.array([1.0])) as sa:
        with pytest.raises(ValueError):
            sa.array[0] = 2.0


def test_shared_strings_roundtrip():
    strings = ["a", "bb", "ccc"]
    with SharedStrings(strings) as ss:
        assert len(ss) == 3
        assert ss[0] == "a"
        assert ss[1] == "bb"
        assert ss[2] == "ccc"
        assert ss[-1] == "ccc"
        assert ss[0:2] == ["a", "bb"]
        with pytest.raises(IndexError):
            _ = ss[3]
        with pytest.raises(IndexError):
            _ = ss[-4]


def test_shared_strings_empty():
    with pytest.raises(ValueError):
        SharedStrings([])


def test_shared_list_roundtrip():
    items = [1, 2.0, "three", (4, 5)]
    with SharedList(items) as sl:
        assert len(sl) == 4
        assert sl[0] == 1
        assert sl[2] == "three"
        assert sl[3] == (4, 5)
        assert sl[-1] == (4, 5)
        assert sl[1:3] == [2.0, "three"]
        with pytest.raises(IndexError):
            _ = sl[4]
        with pytest.raises(IndexError):
            _ = sl[-5]


def test_shared_list_empty():
    with pytest.raises(ValueError):
        SharedList([])


def test_shared_list_allow_pickle_false():
    items = [1, "a", 2.0, None, True]
    with SharedList(items, allow_pickle=False) as sl:
        for i, x in enumerate(items):
            assert sl[i] == x
    with pytest.raises(TypeError):
        SharedList([set()], allow_pickle=False)


def test_handle_size():
    items = list(range(100_000))
    with SharedList(items) as sl:
        assert len(sl) == 100_000
        assert len(pickle.dumps(sl)) < 1024


def test_shared_dict_roundtrip():
    d = {"a": 1, "b": 2}
    with SharedDict(d) as sd:
        assert len(sd) == 2
        assert sd["a"] == 1
        assert sd["b"] == 2
        with pytest.raises(KeyError):
            _ = sd["c"]
        assert set(sd) == {"a", "b"}


def test_shared_dict_empty():
    with pytest.raises(ValueError):
        SharedDict({})


def test_shared_dict_collision(monkeypatch):
    def fake_blake2b(data, digest_size=8):
        class FakeHash:
            def digest(self):
                return b"\x00" * digest_size

        return FakeHash()

    monkeypatch.setattr("dense_armor.utility.data.shared.hashlib.blake2b", fake_blake2b)
    d = {"a": 1, "b": 2}
    with SharedDict(d) as sd:
        assert sd["a"] == 1
        assert sd["b"] == 2


def test_to_shared_array():
    with to_shared(np.array([1, 2])) as sa:
        assert isinstance(sa, SharedArray)


def test_to_shared_strings():
    with to_shared(["a", "b"]) as ss:
        assert isinstance(ss, SharedStrings)


def test_to_shared_list():
    with to_shared([1, 2]) as sl:
        assert isinstance(sl, SharedList)


def test_to_shared_dict():
    with to_shared({"a": 1}) as sd:
        assert isinstance(sd, SharedDict)


def test_to_shared_invalid():
    with pytest.raises(TypeError):
        to_shared(42)


def test_spawn_pool():
    ctx = mp.get_context("spawn")
    items = ["a", "b", "c"]
    with SharedStrings(items) as ss:
        with ctx.Pool(4) as pool:
            results = pool.map(_read_index, [(ss, i) for i in range(3)])
            assert results == items
        assert ss[0] == "a"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")
def test_no_leak():
    shm_dir = Path("/dev/shm")
    if not shm_dir.exists():
        pytest.skip("/dev/shm not found")
    before = set(shm_dir.iterdir())
    with SharedArray(np.array([1.0])):
        pass
    with SharedStrings(["a"]):
        pass
    with SharedList([1]):
        pass
    with SharedDict({"a": 1}):
        pass
    gc.collect()
    after = set(shm_dir.iterdir())
    assert after == before


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")
def test_fork_child_reads():
    ctx = mp.get_context("fork")
    items = ["a", "b", "c"]
    with SharedStrings(items) as ss:
        with ctx.Pool(2) as pool:
            results = pool.map(_read_index, [(ss, i) for i in range(3)])
            assert results == items


def test_shared_memory_size_is_python_int(monkeypatch):
    """SharedMemory(size=...) receives a Python int, never a numpy.int64.

    On Windows CreateFileMapping() raises TypeError for a numpy.int64.
    """
    from multiprocessing import shared_memory

    real = shared_memory.SharedMemory
    sizes = []

    def spy(*args, **kwargs):
        if kwargs.get("create"):
            sizes.append(kwargs["size"])
        return real(*args, **kwargs)

    monkeypatch.setattr(
        "dense_armor.utility.data.shared.shared_memory.SharedMemory", spy
    )
    with SharedArray(np.array([1.0])), SharedStrings(["a"]), SharedList([1]):
        pass
    with SharedDict({"a": 1}):
        pass
    assert len(sizes) == 1 + 2 + 2 + 6
    assert all(type(s) is int for s in sizes)


@pytest.mark.parametrize(
    "make",
    [
        lambda: SharedArray(np.array([1.0, 2.0])),
        lambda: SharedStrings(["a", "b"]),
        lambda: SharedList([1, 2]),
        lambda: SharedDict({"a": 1}),
    ],
    ids=["array", "strings", "list", "dict"],
)
def test_finalizer_frees_the_block(make):
    """Dropping the owner frees the block: the finalizer holds no reference to self."""
    from multiprocessing import shared_memory

    obj = make()
    name = obj.names[0]
    del obj
    gc.collect()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


@pytest.mark.parametrize(
    ("make", "get"),
    [
        (lambda: SharedArray(np.array([1.0, 2.0])), lambda h: h.array.tolist()),
        (lambda: SharedStrings(["a", "bb"]), lambda h: h[:]),
        (lambda: SharedList([1, {"k": 2}]), lambda h: h[:]),
        (lambda: SharedDict({"a": 1, "b": 2}), lambda h: [h["a"], h["b"]]),
    ],
    ids=["array", "strings", "list", "dict"],
)
def test_pickled_handle_attaches_and_never_unlinks(make, get):
    with make() as owner:
        att = pickle.loads(pickle.dumps(owner))
        try:
            assert len(att) == len(owner)
            assert att.names == owner.names
            assert get(att) == get(owner)
            att.unlink()
            att.unlink()
        finally:
            att.close()
        assert get(owner) == get(owner)
        owner.unlink()
        owner.unlink()


def test_shared_list_safe_tuple_and_list_roundtrip():
    items = [(1, "a"), [2.0, b"x", None, True]]
    with SharedList(items, allow_pickle=False) as sl:
        assert sl[:] == items


def test_shared_strings_all_empty_raises():
    with pytest.raises(ValueError):
        SharedStrings(["", ""])


def test_to_shared_empty_list_raises():
    with pytest.raises(ValueError):
        to_shared([])


def test_shared_array_rejects_object_dtype():
    """dtype=object would put pointers in the shared block: rejected."""
    import numpy as np

    with pytest.raises(TypeError):
        SharedArray(np.array([{"a": 1}], dtype=object))


def test_shared_list_allow_pickle_recursive():
    """allow_pickle=False must reject nested unsafe types."""
    with pytest.raises(TypeError):
        SharedList([[{1}]], allow_pickle=False)
    with pytest.raises(TypeError):
        SharedList([(1, {2})], allow_pickle=False)


def test_shared_dict_allow_pickle_recursive():
    """allow_pickle=False must reject unsafe keys and values."""
    with pytest.raises(TypeError):
        SharedDict({1: {2}}, allow_pickle=False)
    with pytest.raises(TypeError):
        SharedDict({{1}: 2}, allow_pickle=False)


def test_shared_dict_key_equality_declared():
    """SharedDict compares keys by pickle bytes, not by Python equality.

    A Python dict treats 1, 1.0 and True as the same key. SharedDict
    treats them as distinct keys. This is a declared limitation, not a
    bug: the docstring of SharedDict says it.
    """
    d = SharedDict({1: "a"})
    try:
        assert d[1] == "a"
        with pytest.raises(KeyError):
            _ = d[1.0]
        with pytest.raises(KeyError):
            _ = d[True]
    finally:
        d.close()
        d.unlink()


class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __getstate__(self):
        return {"x": self.x, "y": self.y}

    def __setstate__(self, state):
        self.x, self.y = state["x"], state["y"]

    def __eq__(self, other):
        return isinstance(other, Point) and (self.x, self.y) == (other.x, other.y)


def test_shared_list_custom_object():
    """A custom class with __getstate__ / __setstate__ survives the round trip."""
    with SharedList([Point(1, 2), Point(3, 4)]) as sl:
        a, b = sl[0], sl[1]
        assert a == Point(1, 2)
        assert b == Point(3, 4)
        assert a is not b


def test_shared_list_rejects_connection():
    """A non-picklable object (a socket) raises TypeError."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(TypeError):
            SharedList([s])
    finally:
        s.close()


def test_shared_list_cyclic_structure():
    """A cyclic structure is picklable, so SharedList round-trips it."""
    a = [1, 2]
    a.append(a)
    with SharedList([a]) as sl:
        got = sl[0]
        assert got[0] == 1 and got[1] == 2 and got[2] is got
