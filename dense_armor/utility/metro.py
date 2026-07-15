# -*- coding: utf-8 -*-
"""
Sentinel Metrology Framework - Logarithmic Scaling Module
=========================================================
Sottosistema: Protezione Differenziale dei Gradienti (Anti-Underflow)
Percorso: sentinel02/utility/metro.py
"""

import numpy as np

class Metro:
    def __init__(self, val_e: float = -4.0):
        """Inizializza lo scaler logaritmico."""
        self.val_e = float(val_e)
        self.eps64 = np.finfo(np.float64).eps

    def enc(self, raw_v: float) -> tuple:
        """Proietta la variabile infinitesima nello spazio hardware visibile."""
        v64 = np.float64(raw_v)
        if v64 <= 0.0:
            return np.float64(0.0), np.float64(1.0)
        
        exp10 = np.log10(v64)
        shift = self.val_e - exp10
        fact = np.float64(10 ** shift)
        v_new = v64 * fact
        return v_new, fact

    def dec(self, v_new: float, fact: float) -> float:
        """Ripristina il valore originale."""
        return np.float64(v_new) / np.float64(fact)
