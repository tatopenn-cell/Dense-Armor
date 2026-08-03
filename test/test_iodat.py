# -*- coding: utf-8 -*-
import h5py
import netCDF4
import numpy as np
import pytest

from dense_armor.utility.iodat import lodat


def test_lodat_file_assente_solleva_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError):
        lodat(str(tmp_path / "assente.h5"), "dataset")


def test_lodat_legge_hdf5_reale(tmp_path):
    path = str(tmp_path / "dati.h5")
    original = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    with h5py.File(path, "w") as f:
        f.create_dataset("tensore", data=original)

    loaded = lodat(path, "tensore")

    np.testing.assert_array_equal(loaded, original)


def test_lodat_legge_hdf5_estensione_hdf5(tmp_path):
    path = str(tmp_path / "dati.hdf5")
    original = np.ones((2, 2))
    with h5py.File(path, "w") as f:
        f.create_dataset("d", data=original)

    loaded = lodat(path, "d")
    np.testing.assert_array_equal(loaded, original)


def test_lodat_legge_netcdf_reale(tmp_path):
    path = str(tmp_path / "dati.nc")
    original = np.arange(12, dtype=np.float64).reshape(3, 4)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as f:
        f.createDimension("x", 3)
        f.createDimension("y", 4)
        var = f.createVariable("tensore", "f8", ("x", "y"))
        var[:, :] = original

    loaded = lodat(path, "tensore")

    np.testing.assert_allclose(loaded, original)


def test_lodat_estensione_non_supportata_solleva_valueerror(tmp_path):
    path = tmp_path / "dati.txt"
    path.write_text("non un dataset")

    with pytest.raises(ValueError):
        lodat(str(path), "qualsiasi")
