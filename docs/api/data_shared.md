# Shared-memory containers

How do I let worker processes read a very large list without every worker copying it?

## 1. See the growth with a plain list

A `multiprocessing.Pool` with the `fork` start method inherits the parent memory. The first time a worker reads an object in that memory, the kernel copies the page that holds it. Every worker copies a page, and the memory of every worker grows.

```python
import multiprocessing as mp
import psutil, os
from dense_armor.utility.data import SharedStrings

data = [f"/path/{i}" for i in range(1_000_000)]
```

## 2. Use `SharedStrings` or `SharedList`

`SharedStrings` stores the strings in one UTF-8 buffer, with an `int64` offsets array. `SharedList` stores the pickled items in one buffer, with the same offsets array. Both live in `/dev/shm` (Linux) or in the `SharedMemory` block (Windows).

```python
shared = SharedStrings(data)
shared[0]
```

## 3. Hand the handle to a `spawn` Pool

A handle can be pickled. The pickle carries only the block names and the lengths, not the data.

```python
import multiprocessing as mp
ctx = mp.get_context("spawn")
with ctx.Pool(4) as pool:
    pass
```

The table of item C (from `benchmarks/data_shared_memory.py`, fraction 1.0, `fork`, 1,000,000 synthetic paths, 4 workers, 5 runs, median and IQR):

| Contender | USS median (B) | USS IQR (B) | PSS median (B) | Latency (µs) |
|---|---:|---:|---:|---:|
| plain list | 113,889,280 | 6,144 | 100,090,880 | 5.85 |
| numpy object array | 114,216,960 | 4,096 | 100,311,040 | 6.17 |
| `Manager().list` | 49,684,480 | 4,096 | 48,901,120 | 152.39 |
| `ShareableList` | 81,960,960 | 4,096 | 72,058,880 | 13.03 |
| `SharedStrings` | 50,012,160 | 12,288 | 45,265,920 | 2.73 |
| `SharedList` | 50,012,160 | 0 | 48,256,000 | 3.71 |

The three claims declared in `ISTRUZIONI.md`, measured on fraction 1.0:

1. **Worker USS of `SharedStrings` and `SharedList` below 0.5 times that of the plain list: holds.** The plain list gives 113,889,280 bytes, `SharedStrings` and `SharedList` give 50,012,160; the ratio is 0.44, below 0.5.
2. **Read latency: holds.** `SharedStrings` reads at 2.73 µs per item against 5.85 µs of the plain list (0.47×), and `SharedList` reads at 3.71 µs against 152.39 µs of `Manager().list` (41× faster).
3. **`spawn` result within 20% of the `fork` result: holds.** `SharedStrings`: fork 50,012,160 bytes, spawn 47,677,440, difference 4.7%. `SharedList`: fork 50,012,160, spawn 47,718,400, difference 4.6%.

## Details

- **References**: the issue pytorch/pytorch#13246 (`papers/issue_13246.md`), the post `papers/ppwwyyxx_demystify_ram.txt` (RSS, PSS, USS), the pull request `papers/pr_191555.md` as prior art, the `multiprocessing` docs.
- **Choices**: offsets and one flat buffer; blake2b digest, not `hash()`; the owner unlinks and an attached handle only closes; read-only views; `allow_pickle`; empty input raises; `uss` and `pss`, not `rss`.
- **Measured cost**: the JSON of `benchmarks/data_shared_memory.py`, 5 runs, median and IQR.
