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
        with ctx.Pool(2) as pool:
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
    gc.collect()
    after = set(shm_dir.iterdir())
    assert len(after - before) == 0
