import numpy as np
import pytest

import dense_armor as da


def test_default_is_float32_and_can_be_changed_back():
    assert da.get_default_dtype() == np.dtype(np.float32)
    try:
        da.set_default_dtype("float64")
        assert da.get_default_dtype() == np.dtype(np.float64)
    finally:
        da.set_default_dtype("float32")
    assert da.get_default_dtype() == np.dtype(np.float32)


@pytest.mark.parametrize("bad", ["float16", "int32", np.int64])
def test_only_float32_and_float64_are_accepted(bad):
    with pytest.raises(ValueError):
        da.set_default_dtype(bad)
