# -*- coding: utf-8 -*-
"""Compatibility shim.

Template definitions now live in prompt_generate/cd_prompt.py so that ASI,
CRAC, and SAN share one unified prompt file. This module only re-exports the
classes to avoid breaking any older imports.
"""

from __future__ import annotations

from .cd_prompt import ObjTaskASI, ObjTaskASIA, ObjTaskCRAC, ObjTaskChild, ObjTaskSAN

__all__ = ["ObjTaskASI", "ObjTaskASIA", "ObjTaskCRAC", "ObjTaskChild", "ObjTaskSAN"]
