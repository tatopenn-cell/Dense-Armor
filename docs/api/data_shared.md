# Shared-memory containers

How do I let worker processes read a very large list without every worker copying it?

## 1. See the growth with a plain list

A `multiprocessing.Pool` with the `fork` start method inherits the parent memory. The first time a worker reads an object in that memory, the kernel copies the page that holds it, because reading a Python object changes its reference count. Every worker copies pages, and the memory of every worker grows.

The benchmark builds 1,000,000 synthetic strings that look like paths, starts 4 workers, and measures the unique set size (`uss`) and the proportional set size (`pss`) of each worker before and after it reads the items. `rss` is not used: it counts the shared pages, so it cannot tell shared from copied. The benchmark runs on Linux:

```bash
python benchmarks/data_shared_memory.py --smoke
```

The table below is the full run (1,000,000 items, 5 runs, median and IQR), from `benchmarks/data_shared_memory.json`. Under `fork` the plain list gives a worker `uss` of 104,611,840 bytes.

## 2. Use `SharedStrings` or `SharedList`

`SharedStrings` stores the strings in one UTF-8 buffer, with an `int64` offsets array; reading item `i` decodes only item `i`. `SharedList` stores the pickled items in one buffer, with the same offsets array; reading item `i` unpickles only item `i`. Both keep the data off the Python heap, in a `SharedMemory` block (`/dev/shm` on Linux).

```python
from dense_armor.utility.data import SharedStrings

paths = [f"/data/img_{i:06d}.png" for i in range(1000)]
ss = SharedStrings(paths)
print(len(ss), ss[0], ss[-1])
ss.unlink()
ss.close()
```

```text
1000 /data/img_000000.png /data/img_000999.png
```

`unlink()` frees the block (only the creator does it); `close()` detaches this process from it.

## 3. Hand the handle to a `spawn` Pool

A handle can be pickled. The pickle carries only the block names and the lengths, not the data, so it stays under 1 KiB however long the list is.

```python
import pickle
from dense_armor.utility.data import SharedStrings

ss = SharedStrings([f"/data/img_{i:06d}.png" for i in range(100_000)])
att = pickle.loads(pickle.dumps(ss))
print(len(pickle.dumps(ss)) < 1024, att[99_999])
att.close()
ss.unlink()
ss.close()
```

```text
True /data/img_099999.png
```

The unpickled handle `att` attaches to the same blocks; closing it never frees them. A `Pool` does the same thing for each worker:

```python
import multiprocessing as mp
from dense_armor.utility.data import SharedStrings


def read(args):
    ss, i = args
    return ss[i]


if __name__ == "__main__":
    with SharedStrings([f"/data/img_{i}.png" for i in range(4)]) as ss:
        with mp.get_context("spawn").Pool(4) as pool:
            print(pool.map(read, [(ss, i) for i in range(4)]))
```

```text
['/data/img_0.png', '/data/img_1.png', '/data/img_2.png', '/data/img_3.png']
```

This is a script, not a notebook cell: `spawn` workers import the file, so `read` must live at module level.

## Results

`fork`, fraction 1.0 (every item read), 1,000,000 synthetic paths, 4 workers, 5 runs, median and IQR, from `benchmarks/data_shared_memory.py`:

| Contender | USS median (B) | USS IQR (B) | PSS median (B) | Latency median (µs) |
|---|---:|---:|---:|---:|
| plain list | 104,611,840 | 4,096 | 91,772,928 | 6.24 |
| numpy object array | 112,594,944 | 8,192 | 99,753,984 | 6.70 |
| `Manager().list` | 48,320,512 | 2,048 | 48,307,200 | 131.86 |
| `ShareableList` | 80,461,824 | 8,192 | 76,823,552 | 13.15 |
| `SharedStrings` | 48,349,184 | 8,192 | 49,253,888 | 2.66 |
| `SharedList` | 48,353,280 | 20,480 | 51,175,936 | 3.67 |

`spawn`, same data (the plain list is passed to each worker as an argument, so each worker holds its own copy):

| Contender | USS median (B) | USS IQR (B) | PSS median (B) | Latency median (µs) |
|---|---:|---:|---:|---:|
| plain list | 46,698,496 | 200,704 | 46,697,984 | 0.64 |
| `ShareableList` | 48,109,568 | 2,048 | 54,135,296 | 5.18 |
| `SharedStrings` | 40,157,184 | 0 | 45,694,464 | 2.74 |
| `SharedList` | 40,157,184 | 0 | 48,697,344 | 3.70 |

The three claims were declared before the run (`ISTRUZIONI.md`, item C) and are read from the two tables:

1. **Worker `uss` of `SharedStrings` and `SharedList` below 0.5 times that of the plain list under `fork`: holds.** 48,349,184 / 104,611,840 = 0.462 and 48,353,280 / 104,611,840 = 0.462.
2. **Read latency: holds, on both parts.** `SharedStrings` is within 10 times of the plain list: 2.66 µs against 6.24 µs under `fork` (0.43), and 4.1 times against the plain list under `spawn` (2.74 µs against 0.64 µs). The plain list reads in 6.24 µs under `fork` and 0.64 µs under `spawn`; the benchmark does not isolate the reason for the difference, so both ratios are given. `SharedList` is at least 10 times faster than `Manager().list`: 131.86 µs against 3.67 µs, 36.0 times.
3. **`spawn` result within 20% of the `fork` result: holds with `fork` as the reference, does not hold with `spawn` as the reference.** The `uss` of `SharedStrings` is 48,349,184 under `fork` and 40,157,184 under `spawn`: the difference is 16.9% of the `fork` value and 20.4% of the `spawn` value. `SharedList` gives 17.0% and 20.4%. The claim did not say which of the two is the reference.

## Details

### References

- Issue pytorch/pytorch#13246, `papers/issue_13246.md`: comment 1364587359 (ppwwyyxx: replacing Python objects with a `numpy` array solves the growth only with the `fork` start method) and comment 436632186 (the original poster: iterating over a huge list makes memory grow, iterating over a `numpy` array does not).
- Post `papers/ppwwyyxx_demystify_ram.txt`: section "Measure RAM Usage" (RSS, PSS, USS) and section "Copy-on-read of forked CPython objects" (the reference-count mechanism).
- Pull request 191555, `papers/pr_191555.md`, as prior art: a flat buffer with `int64` offsets. It is read, not run.
- Python documentation, `multiprocessing.shared_memory.SharedMemory`: the `track` parameter (3.13), and the resource tracker for attached blocks (3.10 to 3.12).

### Choices stated explicitly

- One flat buffer plus an `int64` offsets array, so reading item `i` touches only that item.
- `SharedDict` looks up with `hashlib.blake2b` over the pickle of the key, not `hash()`: the `hash()` of a `str` differs between processes. The digests are sorted in a `uint64` array and found with `numpy.searchsorted`; the stored key is then compared, so a collision never returns the wrong value.
- The creator owns the blocks and unlinks them (at `unlink()`, at the end of a `with`, at garbage collection or at exit). An attached handle only closes. On Linux with Python 3.10 to 3.12 an attached handle is removed from the resource tracker; with Python 3.13 it attaches with `track=False`. On Windows there is no resource tracker, and `unlink()` does nothing: the block disappears when the last handle closes.
- `SharedArray.array` is read-only.
- `allow_pickle=False` restricts what is written at construction to `None`, `bool`, `int`, `float`, `str`, `bytes` and `tuple` or `list` of these. Reading still unpickles, so every local process that can attach the blocks is trusted.
- An empty list or dict raises `ValueError`, and so does a list of strings that are all empty: a `SharedMemory` block cannot have size 0.
- `uss` and `pss` are measured, `rss` is not.

### Declared limitation of `SharedDict`

Keys are compared by the bytes of their pickle, not by Python equality. On a dict with key `1`, `d[1.0]` and `d[True]` raise `KeyError`, while a Python `dict` returns the value. The reason is the cost of a lookup: a digest-only match would unpickle every candidate key, which is slower than the byte comparison.

### What pickles and what does not

Custom classes with `__getstate__` and `__setstate__`, nested dicts, cyclic structures and tensors go through. Connections, file handles, sockets, threads, locks, lambdas and generators are not picklable: put the recipe (the URL, the credentials, the path) in the container and open the resource in the worker.

### Measured cost

The JSON of `benchmarks/data_shared_memory.py`: 1,000,000 synthetic items, 4 workers, 5 runs, median and IQR, all sizes in bytes.

### Development diary

On Windows the first version hung every `spawn` Pool: attaching a handle called `resource_tracker.unregister`, which on Windows tries to start the tracker with `_posixsubprocess`, a module that Windows does not have. The worker failed while unpickling and `Pool.map` never returned. The call now runs only off Windows, and on Linux it uses the registered name (with its leading `/`), which `SharedMemory.name` does not return.
