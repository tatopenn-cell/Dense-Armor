"""Pydantic input schemas for every dense_armor_mcp tool, extracted from
tools.py so tool logic and schema definitions can change independently.
Field descriptions stay here (not moved into tool docstrings) since MCP
clients read them directly off the schema to build their own UI/validation.
"""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CleanSignalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: List[Optional[float]] = Field(
        ..., min_length=1,
        description="Raw 1D series to clean -- NaN allowed (use JSON null for a missing reading).",
    )
    x_reference: Optional[List[Optional[float]]] = Field(
        default=None,
        description="Known-clean reference values, same length as 'values', if you have one. "
        "Omit for blind mode (Orca estimates a causal baseline from the series itself).",
    )
    use_arbiter: bool = Field(
        default=True,
        description="Route each point to the right corrector (isolated spike vs sustained "
        "regime change vs no anomaly) instead of one gate for the whole series. "
        "Verified never worse than the default, better on most anomaly types -- see "
        "dense_armor's utility/arbiter.py.",
    )


class DetectAnomaliesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: List[float] = Field(..., min_length=1, description="1D series to classify (no NaN -- fill gaps first).")
    radius: int = Field(default=10, ge=1, description="Reference window span (in samples) before ref_mult scaling.")
    ref_mult: int = Field(default=3, ge=1, description="Reference window = radius * ref_mult samples, causal (only points before each candidate).")
    n_sigmas: float = Field(default=3.0, gt=0.0, description="Deviation threshold, in scaled MAD units, above which a point is flagged.")
    spike_run_max: int = Field(default=2, ge=1, description="Longest run of consecutive deviant points still classified as an isolated spike; longer runs are checked for a genuine regime change.")


class RobustFilterInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: List[float] = Field(..., min_length=1, description="1D series to filter (no NaN -- fill gaps first).")
    method: str = Field(
        default="pressure_valve",
        description="'pressure_valve' (all 4 methods combined via a Lagrange-derived minimum-variance "
        "estimator, JSD-adaptive threshold) or one of 'chauvenet', 'tukey', 'hampel', 'sigma_clip' alone.",
    )
    radius: int = Field(default=10, ge=1, description="Local window span in samples.")


class HealSeriesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: List[float] = Field(..., min_length=1, description="1D series to heal (no NaN -- fill gaps first).")
    radius: int = Field(default=2, ge=1, description="Narrow window (samples) used for the neighbor-consensus check.")
    sustain_threshold: float = Field(default=0.7, gt=0.0, le=1.0, description="Fraction of neighbors that must share a deviation's sign/magnitude for it to count as a genuine, sustained change.")
    wide_mult: int = Field(default=3, ge=1, description="Wide baseline window = radius * wide_mult samples.")


class StreamStartInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_channels: int = Field(..., ge=1, description="Number of independent channels (e.g. robot joints, IMU axes) this session will receive one real-time reading per, per update call.")
    radius: int = Field(default=10, ge=1, description="Causal reference window span (in samples) before ref_mult scaling -- same convention as dense_armor_detect_anomalies.")
    ref_mult: int = Field(default=3, ge=1, description="Reference window = radius * ref_mult samples.")
    n_sigmas: float = Field(default=3.0, gt=0.0, description="Deviation threshold, in scaled MAD units, above which a point is flagged.")


class StreamUpdateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(..., min_length=1, description="Session id returned by dense_armor_stream_start.")
    values: List[float] = Field(..., min_length=1, description="One real-time reading per channel (same order, same length as n_channels given to dense_armor_stream_start) -- no NaN.")


class StreamEndInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(..., min_length=1, description="Session id to close and free -- call this when a real sensor stream ends, sessions are not garbage-collected automatically.")
