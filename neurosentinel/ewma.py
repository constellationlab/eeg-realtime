"""Stateful online exponentially weighted moving average."""

from __future__ import annotations


class EWMA:
    """y_t = beta * y_{t-1} + (1 - beta) * x_t.  Seeds from the first observation."""

    def __init__(self, beta: float = 0.9) -> None:
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        self._beta = beta
        self._value: float | None = None

    def update(self, x: float) -> float:
        """Ingest a new observation and return the updated smoothed value."""
        if self._value is None:
            self._value = x
        else:
            self._value = self._beta * self._value + (1.0 - self._beta) * x
        return self._value

    @property
    def current(self) -> float | None:
        """Current smoothed value, or None if never updated."""
        return self._value

    def reset(self) -> None:
        """Reset to uninitialised state."""
        self._value = None
