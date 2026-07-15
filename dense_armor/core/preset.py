# -*- coding: utf-8 -*-

SENTINEL_PRESETS = {
    "balanced_v2": {
        "static_threshold": 1e-05,
        "initial_damping": 0.05,
        "alpha": 0.05,
        "anomaly_sigma_mult": 1.0,
        "k_anom_min": 0.18,
        "k_anom_max": 0.28
    },
    "cifar10_best_v1": {
        "static_threshold": 4.384205263047182e-06,
        "initial_damping": 0.05469714496620124,
        "alpha": 0.04590571010729771,
        "anomaly_sigma_mult": 0.8,
        "k_anom_min": 0.19814386535420334,
        "k_anom_max": 0.2945784399322787
    },
    "pure_1d_time_v1": {
        "static_threshold": 0.08,
        "initial_damping": 0.15,
        "alpha": 0.35,
        "anomaly_sigma_mult": 1.2,
        "k_anom_min": 0.25,
        "k_anom_max": 0.55
    },
    # =====================================================================
    # NUOVO PRESET CALIBRATO: SCUDO DI LYAPUNOV ATTIVO PER AUDIT 2D
    # =====================================================================
    "cifar10_hardened_lyapunov": {
        "static_threshold": 0.12,
        "initial_damping": 0.60,
        "alpha": 0.05,
        "anomaly_sigma_mult": 1.0,
        "k_anom_min": 0.18,
        "k_anom_max": 0.28
    }
}

# -*- coding: utf-8 -*-

SENTINEL_PRESETS = {
    "balanced_v2": {
        "static_threshold": 1e-05,
        "initial_damping": 0.05,
        "alpha": 0.05,
        "anomaly_sigma_mult": 1.0,
        "k_anom_min": 0.18,
        "k_anom_max": 0.28
    },
    "cifar10_best_v1": {
        "static_threshold": 4.384205263047182e-06,
        "initial_damping": 0.05469714496620124,
        "alpha": 0.04590571010729771,
        "anomaly_sigma_mult": 0.8,
        "k_anom_min": 0.19814386535420334,
        "k_anom_max": 0.2945784399322787
    },
    "pure_1d_time_v1": {
        "static_threshold": 0.08,
        "initial_damping": 0.15,
        "alpha": 0.35,
        "anomaly_sigma_mult": 1.2,
        "k_anom_min": 0.25,
        "k_anom_max": 0.55
    },
    # =====================================================================
    # NUOVO PRESET CALIBRATO: SCUDO DI LYAPUNOV ATTIVO PER AUDIT 2D
    # =====================================================================
    "cifar10_hardened_lyapunov": {
        "static_threshold": 0.12,
        "initial_damping": 0.60,
        "alpha": 0.05,
        "anomaly_sigma_mult": 1.0,
        "k_anom_min": 0.18,
        "k_anom_max": 0.28
    }
}

