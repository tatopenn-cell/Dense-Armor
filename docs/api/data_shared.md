# Shared-memory containers

How do I let worker processes read a very large list without every worker copying it?

## 1. See the growth with a plain list

A `multiprocessing.Pool` with the `fork` start method inherits the parent memory. The first time a worker reads an object in that memory, the kernel copies the page that holds it. Every worker copies a page, and the memory of every worker grows.

```python
import multiprocessing as mp
import psutil, os
from dense_armor.utility.data import SharedStrings

data = [f"/path/{i}" for i in range(1_000_000)]
