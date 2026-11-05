"""Tests for RingBuffer."""

import threading

import numpy as np
import pytest

from dense_armor.utility.finance.ring import RingBuffer


def test_fifo_order():
    b = RingBuffer(4, 2)
    for i in range(3):
        assert b.push(np.array([float(i), float(i * 10)]))
    out = np.zeros(2)
    for i in range(3):
        assert b.pop_into(out)
        assert out.tolist() == [float(i), float(i * 10)]
    assert len(b) == 0


def test_push_on_full_returns_false_and_leaves_data():
    b = RingBuffer(2, 1)
    assert b.push(np.array([1.0]))
    assert b.push(np.array([2.0]))
    assert b.push(np.array([3.0])) is False
    assert len(b) == 2
    out = np.zeros(1)
    assert b.pop_into(out) and out[0] == 1.0
    assert b.pop_into(out) and out[0] == 2.0


def test_pop_on_empty_returns_false_and_leaves_out():
    b = RingBuffer(2, 2)
    out = np.full(2, 9.0)
    assert b.pop_into(out) is False
    assert out.tolist() == [9.0, 9.0]


def test_wrap_around():
    b = RingBuffer(4, 1)
    out = np.zeros(1)
    n = 3 * 4 + 1
    for i in range(n):
        assert b.push(np.array([float(i)]))
        assert b.pop_into(out)
        assert out[0] == float(i)
    assert len(b) == 0


def test_push_stores_a_copy():
    b = RingBuffer(2, 1)
    row = np.array([1.0])
    assert b.push(row)
    row[0] = 99.0
    out = np.zeros(1)
    assert b.pop_into(out)
    assert out[0] == 1.0


def test_constructor_rejects_bad_sizes():
    with pytest.raises(ValueError):
        RingBuffer(0, 1)
    with pytest.raises(ValueError):
        RingBuffer(1, 0)


def test_push_rejects_wrong_shape():
    b = RingBuffer(2, 3)
    with pytest.raises(ValueError):
        b.push(np.zeros(2))
    with pytest.raises(ValueError):
        b.push(np.zeros((1, 3)))


def test_pop_into_rejects_wrong_shape():
    b = RingBuffer(2, 3)
    with pytest.raises(ValueError):
        b.pop_into(np.zeros(2))


def test_two_threads_50k_rows_in_order():
    b = RingBuffer(64, 1)
    out = np.zeros(1)
    n = 50_000
    received: list[int] = []

    def producer():
        for i in range(n):
            row = np.array([float(i)])
            while not b.push(row):
                pass

    def consumer():
        while len(received) < n:
            if b.pop_into(out):
                received.append(int(out[0]))

    tp = threading.Thread(target=producer)
    tc = threading.Thread(target=consumer)
    tp.start()
    tc.start()
    tp.join(timeout=30.0)
    tc.join(timeout=30.0)
    assert len(received) == n
    assert received == list(range(n))


def test_push_accepts_a_list_and_checks_its_shape():
    b = RingBuffer(2, 3)
    with pytest.raises(ValueError):
        b.push([1.0, 2.0])
    assert b.push([1.0, 2.0, 3.0])
    out = np.zeros(3)
    assert b.pop_into(out)
    assert out.tolist() == [1.0, 2.0, 3.0]
