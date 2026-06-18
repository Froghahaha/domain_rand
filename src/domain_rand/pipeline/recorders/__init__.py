"""Pluggable HDF5 recorders for IL demo collection.

Built-in:
    SimpleRecorder    — flexible dict-based format (default)
    RobomimicRecorder — robomimic-compatible format
    BaseRecorder      — ABC for custom formats
"""

from domain_rand.pipeline.recorders.base import BaseRecorder
from domain_rand.pipeline.recorders.simple import SimpleRecorder
from domain_rand.pipeline.recorders.robomimic import RobomimicRecorder

__all__ = ["BaseRecorder", "SimpleRecorder", "RobomimicRecorder"]
