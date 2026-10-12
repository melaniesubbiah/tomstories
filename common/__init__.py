"""Shared utilities for the ToMStories scripts.

Submodules are imported directly (e.g. ``from common.models import MODELS``) so
that scripts without the heavy dependencies — ``common.hf`` needs torch and
transformers — are not forced to install them.
"""
