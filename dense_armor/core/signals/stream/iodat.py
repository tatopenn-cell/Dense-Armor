"""Read a named tensor out of an HDF5 or NetCDF file.

The two formats are the common containers for scientific arrays, and a
pipeline that accepts either one usually ends up branching on the file
extension in a dozen places. ``lodat`` does the branch once, in one
function, and returns a NumPy array.

The readers are optional: ``h5py`` and ``netCDF4`` are installed with
``pip install dense-armor[data]``. If the reader for a given format is
not installed and the caller opens that format, the module-level
``ImportError`` is raised with a message that names the missing package.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def lodat(fpath: str, dname: str) -> np.ndarray:
    """Read the dataset ``dname`` from the file at ``fpath``.

    Args:
        fpath: path to the file. The extension decides the reader:
            ``.h5`` or ``.hdf5`` for HDF5, ``.nc`` or ``.netcdf`` for
            NetCDF.
        dname: the name of the dataset (HDF5) or of the variable
            (NetCDF) to extract.

    Returns:
        The tensor as a NumPy array, with the shape the file declares.

    Raises:
        FileNotFoundError: if ``fpath`` does not exist.
        ValueError: if the extension is not one of the four accepted.
        ImportError: if the reader for the format is not installed. The
            message names the missing package (`h5py` or `netCDF4`).
    """
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"file not found: {fpath}")

    exten = os.path.splitext(fpath)[1].lower()

    if exten in (".h5", ".hdf5"):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required to read HDF5 files; install it with "
                "`pip install dense-armor[data]`"
            ) from exc
        with h5py.File(fpath, "r") as f:
            data = np.array(f[dname])
        logger.info("HDF5 read: %s | shape: %s", fpath, data.shape)
        return data

    if exten in (".nc", ".netcdf"):
        try:
            import netCDF4
        except ImportError as exc:
            raise ImportError(
                "netCDF4 is required to read NetCDF files; install it with "
                "`pip install dense-armor[data]`"
            ) from exc
        with netCDF4.Dataset(fpath, "r") as f:
            data = np.array(f.variables[dname][:])
        logger.info("NetCDF read: %s | shape: %s", fpath, data.shape)
        return data

    raise ValueError(f"unsupported file format: {exten}")
