"""Random distribution definitions and samplers for domain randomization."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Distribution(ABC):
    """Abstract base class for all randomization distributions."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> Any:
        """Draw a sample from this distribution."""


class Uniform(Distribution):
    """Uniform distribution over [low, high]."""

    def __init__(self, low: float, high: float):
        if low > high:
            raise ValueError(f"Uniform: low ({low}) must be <= high ({high})")
        self.low = low
        self.high = high

    def sample(self, rng: np.random.Generator) -> float:
        return rng.uniform(self.low, self.high)

    def __repr__(self) -> str:
        return f"Uniform(low={self.low}, high={self.high})"


class LogUniform(Distribution):
    """Log-uniform distribution over [low, high] in linear space.

    Samples uniformly in log-space, i.e. P(x) ∝ 1/x for x > 0.
    """

    def __init__(self, low: float, high: float):
        if low <= 0 or high <= 0:
            raise ValueError(f"LogUniform: bounds must be > 0, got [{low}, {high}]")
        if low > high:
            raise ValueError(f"LogUniform: low ({low}) must be <= high ({high})")
        self.low = low
        self.high = high

    def sample(self, rng: np.random.Generator) -> float:
        log_low = np.log(self.low)
        log_high = np.log(self.high)
        return np.exp(rng.uniform(log_low, log_high))

    def __repr__(self) -> str:
        return f"LogUniform(low={self.low}, high={self.high})"


class Normal(Distribution):
    """Normal (Gaussian) distribution with given mean and standard deviation."""

    def __init__(self, mean: float, std: float):
        if std < 0:
            raise ValueError(f"Normal: std ({std}) must be >= 0")
        self.mean = mean
        self.std = std

    def sample(self, rng: np.random.Generator) -> float:
        return rng.normal(self.mean, self.std)

    def __repr__(self) -> str:
        return f"Normal(mean={self.mean}, std={self.std})"


class Choice(Distribution):
    """Categorical distribution — randomly pick one element from a list."""

    def __init__(self, options: list):
        if len(options) == 0:
            raise ValueError("Choice: options list must not be empty")
        self.options = options

    def sample(self, rng: np.random.Generator) -> Any:
        idx = rng.integers(0, len(self.options))
        return self.options[idx]

    def __repr__(self) -> str:
        return f"Choice(options={self.options})"


# Registry for deserialization from config
_DISTRIBUTION_REGISTRY: dict[str, type[Distribution]] = {
    "uniform": Uniform,
    "log_uniform": LogUniform,
    "loguniform": LogUniform,
    "normal": Normal,
    "gaussian": Normal,
    "choice": Choice,
}


def distribution_from_dict(d: dict) -> Distribution:
    """Create a Distribution instance from a configuration dictionary.

    Expected format:
        {"type": "uniform", "low": 0.2, "high": 1.0}
        {"type": "normal", "mean": 0.0, "std": 0.1}
        {"type": "choice", "options": [1, 2, 3]}
    """
    dist_type = d["type"].lower()
    cls = _DISTRIBUTION_REGISTRY.get(dist_type)
    if cls is None:
        raise ValueError(
            f"Unknown distribution type '{dist_type}'. "
            f"Available: {list(_DISTRIBUTION_REGISTRY.keys())}"
        )

    # Remove 'type' key, pass remaining as kwargs
    kwargs = {k: v for k, v in d.items() if k != "type"}
    return cls(**kwargs)
