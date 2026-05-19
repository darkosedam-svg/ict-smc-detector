"""ict-smc-detector — ICT/SMC pattern detection for algorithmic trading.

Public API
----------
detect_fvgs(df, *, track_mitigation=True) -> list[FVG]
detect_bos(df, *, lookback=20, min_break_bps=0.0) -> list[BoS]
detect_order_blocks(df, *, bos_lookback=20, ...) -> list[OrderBlock]
filter_consecutive_bos(events) -> list[BoS]
filter_fvgs(fvgs, *, type, only_unmitigated, min_height, formed_after) -> list[FVG]

Types: FVG, BoS, OrderBlock
"""

from .bos import detect_bos, filter_consecutive_bos
from .fvg import detect_fvgs, filter_fvgs
from .order_block import detect_order_blocks
from .types import BoS, FVG, OrderBlock

__all__ = [
    # Detection functions
    "detect_bos",
    "detect_fvgs",
    "detect_order_blocks",
    # Filter functions
    "filter_consecutive_bos",
    "filter_fvgs",
    # Types
    "BoS",
    "FVG",
    "OrderBlock",
]

__version__ = "0.1.0"
