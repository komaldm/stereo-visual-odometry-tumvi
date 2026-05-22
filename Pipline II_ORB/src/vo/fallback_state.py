from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FallbackState:
    consecutive_pnp_failures: int = 0
    using_vo_fallback: bool = False
    tracking_lost: bool = False
    pending_rebuild: bool = False
