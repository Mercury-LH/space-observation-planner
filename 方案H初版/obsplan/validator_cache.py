"""Bounded, exact Skyfield Earth-orientation reuse for independent validation.

This cache never reads planning windows or approximates/interpolates geometry.
It stores Skyfield 1.55's own ``Time.M`` and ``Time.gast`` at requested seconds.
Large IAU2000A computations are chunked to bound temporary working memory.
The validator's reference backend remains the fallback and the default.
"""
from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import json

import numpy as np

from .config import Config
from .data import Dataset


SUPPORTED_SKYFIELD_VERSION = "1.55"


def _near_shadow_boundary(position_m: np.ndarray, sun_m: np.ndarray) -> bool:
    """Flag a solar ray within one metre of Earth's spherical tangent.

    This only selects the reference fallback; ``is_sunlit`` remains the actual
    shadow test. The conservative tolerance greatly exceeds round-off from the
    exact rotation cache and cannot make a failed observation become accepted.
    """
    from skyfield.constants import ERAD
    earth_m = -position_m
    ray = sun_m + earth_m
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = ray / np.linalg.norm(ray, axis=0)
        projection = np.einsum("ij,ij->j", earth_m, unit)
        closest = np.linalg.norm(earth_m - unit * projection, axis=0)
    return bool(not (np.isfinite(projection).all() and np.isfinite(closest).all())
                or ((projection >= -1.0) & (np.abs(closest - ERAD) <= 1.0)).any())


def _identity(cfg: Config, ds: Dataset) -> str:
    """Bind every configuration/data field, including changes made in place."""
    h = hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True, ensure_ascii=False).encode())
    for field in fields(Dataset):
        h.update(field.name.encode())
        value = getattr(ds, field.name, None)
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                h.update(json.dumps(value.tolist(), ensure_ascii=False).encode())
            else:
                h.update(str(value.dtype).encode())
                h.update(str(value.shape).encode())
                h.update(np.ascontiguousarray(value).tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


class SkyfieldGeometryCache:
    """A scene-local cache reusable for its grid and refined plan checks.

    Construct with the exact ``cfg`` and ``ds`` passed to ``validate``. Content
    changes are rejected. No ephemeris or matrix is loaded until a structurally
    valid, eligible schedule reaches geometry validation. This object is not a
    cross-process or concurrently mutable cache.
    """

    def __init__(self, cfg: Config, ds: Dataset, *, chunk_size: int = 4096,
                 max_points: int = 1_000_000):
        for name, value in (("chunk_size", chunk_size), ("max_points", max_points)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._identity = _identity(cfg, ds)
        self.chunk_size = chunk_size
        self.max_points = max_points
        self._seconds = np.empty(0, np.float64)
        self._matrix = np.empty((3, 3, 0), np.float64)
        self._gast = np.empty(0, np.float64)
        self._computed = 0
        self._reused = 0
        self._chunks = 0
        self._optimized_calls = 0
        self._fallbacks: dict[str, int] = {}

    def assert_compatible(self, cfg: Config, ds: Dataset) -> None:
        if _identity(cfg, ds) != self._identity:
            raise ValueError("SkyfieldGeometryCache is bound to a different cfg or dataset")

    def fallback(self, reason: str) -> None:
        self._fallbacks[reason] = self._fallbacks.get(reason, 0) + 1

    def eligible(self, cfg: Config, start: np.ndarray, duration: np.ndarray,
                 step_s: float) -> bool:
        import skyfield
        if skyfield.__version__ != SUPPORTED_SKYFIELD_VERSION:
            self.fallback("unsupported_skyfield_version")
            return False
        if cfg.moon_sep_deg > 0:
            self.fallback("moon_reference_backend")
            return False
        if step_s != 1.0 or not (np.equal(start, np.floor(start)).all()
                                 and np.equal(duration, np.floor(duration)).all()):
            self.fallback("noninteger_or_subsecond_reference_backend")
            return False
        return True

    def prepare(self, task_seconds: list[np.ndarray], raw_times) -> bool:
        requested = np.unique(np.concatenate(task_seconds))
        if not np.isfinite(requested).all():
            raise ValueError("non-finite validation sample time")
        missing = np.setdiff1d(requested, self._seconds, assume_unique=True)
        if len(self._seconds) + len(missing) > self.max_points:
            self.fallback("point_budget_reference_backend")
            return False
        matrices = np.empty((3, 3, len(missing)), np.float64)
        gast = np.empty(len(missing), np.float64)
        for lo in range(0, len(missing), self.chunk_size):
            hi = min(lo + self.chunk_size, len(missing))
            times = raw_times(missing[lo:hi])
            # Both properties use the same exact IAU2000A values within a chunk.
            gast[lo:hi] = times.gast
            matrices[:, :, lo:hi] = times.M
            self._chunks += 1
        if len(missing):
            seconds = np.r_[self._seconds, missing]
            order = np.argsort(seconds)
            self._seconds = seconds[order]
            self._matrix = np.concatenate((self._matrix, matrices), axis=2)[:, :, order]
            self._gast = np.r_[self._gast, gast][order]
        self._computed += len(missing)
        self._reused += len(requested) - len(missing)
        self._optimized_calls += 1
        return True

    def times(self, seconds: np.ndarray, raw_times):
        index = np.searchsorted(self._seconds, seconds)
        if (index >= len(self._seconds)).any() or not np.array_equal(self._seconds[index], seconds):
            raise ValueError("requested validation time was not prepared in the cache")
        result = raw_times(seconds)
        # Skyfield's cached properties support assignment. Pinned-version tests
        # compare this path with the untouched reference implementation.
        result.gast = self._gast[index]
        result.M = self._matrix[:, :, index]
        return result

    def stats(self) -> dict:
        return dict(backend="skyfield_exact_orientation_cache_v1",
                    skyfield_version=SUPPORTED_SKYFIELD_VERSION,
                    chunk_size=self.chunk_size, max_points=self.max_points,
                    cached_points=len(self._seconds), computed_points=self._computed,
                    reused_points=self._reused, computed_chunks=self._chunks,
                    optimized_calls=self._optimized_calls,
                    reference_fallbacks=sum(self._fallbacks.values()),
                    fallback_reasons=dict(self._fallbacks))
