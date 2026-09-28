# ict-smc-detector

[![CI](https://github.com/darkosedam-svg/ict-smc-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/darkosedam-svg/ict-smc-detector/actions/workflows/ci.yml)

Detection of Fair Value Gaps, Order Blocks, and Break of Structure on OHLCV data. Pure detection — no strategy logic, no entry/exit signals, no opinions about how you should trade them.

The raw FVG detection pass is vectorized (pandas boolean masks over shifted columns). Mitigation tracking, Break-of-Structure, and Order Block detection walk the bar series with per-bar Python loops instead — correct and O(n)-ish, but not vectorized numpy. Tested. Drop into any backtest framework or trading system as the detection layer.

## Why this exists

Most public Python implementations of ICT/SMC concepts are either broken (lookahead bias, wrong rounding), over-complicated (1000-line strategy frameworks), or hidden behind paywalls. This library is the opposite — focused, correct, and small.

Detection should be a commodity. The edge in trading these concepts is in *how* you trade them — context, confluence, risk management. By open-sourcing the detection layer, every trader and bot builder can spend their time on the strategy work that actually moves the needle.

## Install

Not on PyPI yet. Install from GitHub:

```bash
pip install git+https://github.com/darkosedam-svg/ict-smc-detector.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/darkosedam-svg/ict-smc-detector
cd ict-smc-detector
pip install -e .
```

Requires Python 3.10+.

## Quick start

```python
import pandas as pd
from ict_smc import detect_fvgs, detect_order_blocks, detect_bos

# Your OHLCV DataFrame: columns [open, high, low, close], DatetimeIndex
df = pd.read_csv("BTC-1h.csv", index_col="timestamp", parse_dates=True)

# Detect all FVGs with mitigation tracking
fvgs = detect_fvgs(df)
for fvg in fvgs[:5]:
    print(f"{fvg.type} FVG at {fvg.formed_at}: [{fvg.bottom}, {fvg.top}]"
          f"{' (mitigated)' if fvg.is_mitigated else ''}")

# Detect Order Blocks
obs = detect_order_blocks(df, bos_lookback=20)

# Detect Break of Structure
bos_events = detect_bos(df, lookback=20, min_break_bps=10)
```

## What's detected

| Concept | Function | What it returns |
|---|---|---|
| Fair Value Gap | `detect_fvgs(df)` | List of `FVG` objects with type, zone, and mitigation status |
| Order Block | `detect_order_blocks(df)` | List of `OrderBlock` objects (last opposite candle before BoS) |
| Break of Structure | `detect_bos(df)` | List of `BoS` events with direction and broken level |

FVG and Order Block both include mitigation tracking — the timestamp price first re-entered the zone after formation. BoS does not; a break either happened or it didn't, so there's no zone to re-enter. Mitigation is critical for backtesting: an unmitigated FVG from 3 weeks ago is a different signal from one that just formed.

## Filtering

Common post-detection filters built-in:

```python
from ict_smc import filter_fvgs, filter_consecutive_bos

# Only unmitigated bullish FVGs with a minimum height
significant = filter_fvgs(
    fvgs,
    type="bullish",
    only_unmitigated=True,
    min_height=df["close"].mean() * 0.001,  # 10 bps minimum
)

# During strong trends, BoS fires repeatedly. Collapse to last-of-run.
regime_changes = filter_consecutive_bos(bos_events)
```

## Visualization

Not shipped yet. There is no `[viz]` extra and no `examples/` directory in
this repo despite an earlier version of this README describing one — that
was aspirational, not real. The `FVG`, `OrderBlock`, and `BoS` objects are
plain dataclasses (see `ict_smc/types.py`), so plotting them with
matplotlib or plotly yourself is straightforward, just not included.

## Performance

Rough complexity, not benchmarked numbers (the previous version of this
README quoted specific millisecond figures on unspecified hardware that
were never actually measured for this codebase — removed rather than
repeated):

| Operation | Implementation | Complexity |
|---|---|---|
| FVG formation detection | Vectorized pandas boolean masks | O(n) |
| FVG mitigation tracking | Per-bar Python loop over open FVGs | O(n) typical, worse if many FVGs stay open simultaneously |
| BoS detection | Per-bar Python loop computing a rolling max/min slice | O(n · lookback) |
| Order Block detection | Runs BoS, then a bounded backward scan per BoS event | O(n · lookback) plus O(max_search_back) per BoS event |

If you need this fast at large bar counts, the rolling max/min in `bos.py`
is a good candidate to replace with `pandas.Series.rolling(...).max()` —
it isn't vectorized today.

## What this library is NOT

- ❌ Not a trading strategy
- ❌ Not an indicator collection
- ❌ Not a backtest framework
- ❌ Not a charting library

It detects three things correctly and exposes them as Python objects. That's it.

For strategy implementation on top, pair with:
- A backtest framework (vectorbt, backtrader)
- An execution layer (e.g., [`hyperliquid-execution-toolkit`](https://github.com/darkosedam-svg/hyperliquid-execution-toolkit) for crypto perps — note that toolkit's order-placement methods are themselves still stubs)
- Your own signal logic

## Common gotchas

**1. Lookahead bias.** This library carefully avoids it. The BoS detector computes each bar's rolling extreme from an explicit slice of the *prior* `lookback` bars (`highs[i - lookback : i]`, excluding bar `i` itself), so each bar is compared against prior bars only. If you build extensions, preserve this property.

**2. Multiple FVGs in tight ranges.** During fast moves, multiple overlapping FVGs can form in 3-bar windows. The library returns all of them. Whether to merge or pick deepest is a strategy decision — handle it downstream.

**3. BoS in chop.** With small `lookback` values and zero `min_break_bps`, BoS fires constantly during ranging markets. Either raise lookback to 50+, set `min_break_bps=10` or higher, or use `filter_consecutive_bos` to collapse runs.

**4. OB candle vs OB-to-BoS range.** Some implementations define the OB zone as the entire range from OB candle to BoS bar. We use just the OB candle's high/low, which is tighter and more conservative. Easy to extend if you prefer the looser zone.

**5. Pivot-based BoS.** This library uses rolling-extreme BoS, which is fast and good enough for most uses. For more sophisticated BoS based on actual swing pivots (5-bar pivots, etc.), build on top — the existing API is small and easy to extend.

## Contributing

PRs welcome. The bar is "would this detection match what an experienced ICT trader sees on the chart?" If you find a case where the library disagrees with reasonable consensus, open an issue with the OHLCV data and expected behavior.

```bash
# Run the test suite
git clone https://github.com/darkosedam-svg/ict-smc-detector
cd ict-smc-detector
pip install -e .[dev]
pytest tests/
```

22 tests should pass.

## Hire me

I build and harden trading infrastructure: execution engines, exchange connectors, backtesting pipelines, and pattern-detection layers. Available for custom work and ongoing retainers around trading-infrastructure, execution, and backtesting engineering.

Contact: jessuskrist84@gmail.com

## License

MIT. Use it for anything.

## Who built this

Darko Vlahovic — independent algo-trading systems engineer. Available for paid work — strategy implementation, custom detection extensions, full system builds.

- 🌐 [github.com/darkosedam-svg](https://github.com/darkosedam-svg)
- ✉️ [Email](mailto:jessuskrist84@gmail.com)

---

⚠️ **Disclaimer:** This library detects patterns in price data. Pattern detection is not a trading edge by itself. Test thoroughly on historical data and on testnet before risking capital. The MIT license disclaims all warranties — that includes any implication that ICT/SMC concepts predict price.
