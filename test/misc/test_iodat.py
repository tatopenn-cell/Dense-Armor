"""Tests for the HDF5 / NetCDF loader."""

import numpy as np
import pytest

pytest.importorskip("h5py", reason="h5py is optional (dense-armor[data])")

import h5py  # noqa: E402

from dense_armor.utility.misc.iodat import lodat  # noqa: E402


def test_read_h5(tmp_path):
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("temperature", data=np.arange(12).reshape(3, 4))
    out = lodat(str(path), "temperature")
    assert out.shape == (3, 4)
    assert int(out[0, 0]) == 0
    assert int(out[2, 3]) == 11


def test_read_hdf5_extension(tmp_path):
    path = tmp_path / "data.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.array([1.0, 2.0, 3.0]))
    out = lodat(str(path), "x")
    assert out.shape == (3,)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        lodat(str(tmp_path / "missing.h5"), "x")


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="unsupported file format"):
        lodat(str(path), "x")


def test_missing_dataset_raises(tmp_path):
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.array([1.0]))
    with pytest.raises(KeyError):
        lodat(str(path), "does_not_exist")


def test_netcdf_read_if_available(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 is optional")
    path = tmp_path / "data.nc"
    with netCDF4.Dataset(path, "w") as f:
        f.createDimension("n", 5)
        v = f.createVariable("x", "f8", ("n",))
        v[:] = np.arange(5)
    out = lodat(str(path), "x")
    assert out.shape == (5,)
    assert int(out[4]) == 4
