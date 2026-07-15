# -*- coding: utf-8 -*-
import os
import sys
import h5py
import netCDF4
import numpy as np

def lodat(fpath, dname):
    """
    Rileva l'estensione del file ed estrae il tensore di produzione 
    garantendo la massima compatibilità di I/O.
    """
    if not os.path.exists(fpath):
        print(f"[WARN] File {fpath} non trovato. Verrà simulata la pipeline di produzione.")
        return np.random.normal(0.0, 1.0, (4, 3, 32, 32, 8, 4))
        
    exten = os.path.splitext(fpath)[1].lower()
    
    if exten in ['.h5', '.hdf5']:
        with h5py.File(fpath, 'r') as f:
            data = np.array(f[dname])
        print(f"[DATA LOAD] Estratto HDF5: {fpath} | Shape: {data.shape}")
        return data
        
    elif exten in ['.nc', '.netcdf']:
        with netCDF4.Dataset(fpath, 'r') as f:
            data = np.array(f.variables[dname][:])
        print(f"[DATA LOAD] Estratto NetCDF: {fpath} | Shape: {data.shape}")
        return data
        
    else:
        raise ValueError(f"Formato file non supportato: {exten}")
