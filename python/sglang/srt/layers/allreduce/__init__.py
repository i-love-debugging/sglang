"""
AllReduce adaptive configuration system for SGLang.

This module provides adaptive allreduce backend selection based on batch size and hidden size.
"""

from .config import (
    AllReduceBackendConfig,
    get_allreduce_configs,
    get_default_allreduce_config,
    save_allreduce_configs,
    select_allreduce_config,
)
from .adaptive_allreduce import AdaptiveAllReduceLayer
from .manager import (
    get_adaptive_allreduce_layer,
    cleanup_adaptive_allreduce,
)

__all__ = [
    "AllReduceBackendConfig",
    "get_allreduce_configs",
    "get_default_allreduce_config",
    "save_allreduce_configs",
    "select_allreduce_config",
    "AdaptiveAllReduceLayer",
    "get_adaptive_allreduce_layer",
    "cleanup_adaptive_allreduce",
]
