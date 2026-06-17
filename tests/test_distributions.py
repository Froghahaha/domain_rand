"""Tests for the distributions module."""

import numpy as np
import pytest
from domain_rand.core.distributions import (
    Uniform,
    LogUniform,
    Normal,
    Choice,
    distribution_from_dict,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestUniform:
    def test_sample_range(self, rng):
        dist = Uniform(0.0, 1.0)
        samples = [dist.sample(rng) for _ in range(1000)]
        assert all(0.0 <= s <= 1.0 for s in samples)

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            Uniform(1.0, 0.0)


class TestLogUniform:
    def test_sample_range(self, rng):
        dist = LogUniform(0.1, 10.0)
        samples = [dist.sample(rng) for _ in range(1000)]
        assert all(0.1 <= s <= 10.0 for s in samples)

    def test_non_positive_bounds(self):
        with pytest.raises(ValueError):
            LogUniform(-1.0, 1.0)


class TestNormal:
    def test_sample(self, rng):
        dist = Normal(0.0, 0.1)
        s = dist.sample(rng)
        assert isinstance(s, float)


class TestChoice:
    def test_sample_in_options(self, rng):
        dist = Choice(["a", "b", "c"])
        samples = [dist.sample(rng) for _ in range(100)]
        assert all(s in ["a", "b", "c"] for s in samples)

    def test_empty_options(self):
        with pytest.raises(ValueError):
            Choice([])


class TestDistributionFromDict:
    def test_uniform(self):
        dist = distribution_from_dict({"type": "uniform", "low": 0.0, "high": 1.0})
        assert isinstance(dist, Uniform)

    def test_unknown_type(self):
        with pytest.raises(ValueError):
            distribution_from_dict({"type": "garbage"})
