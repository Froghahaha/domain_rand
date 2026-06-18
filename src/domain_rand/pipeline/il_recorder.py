"""Backward-compatible re-export of SimpleRecorder.

Prefer importing from domain_rand.pipeline.recorders directly in new code:
    from domain_rand.pipeline.recorders import SimpleRecorder
"""

from domain_rand.pipeline.recorders.simple import SimpleRecorder as ILRecorder

__all__ = ["ILRecorder"]
