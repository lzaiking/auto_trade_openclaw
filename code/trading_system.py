#!/usr/bin/env python3
"""Run the monthly prediction-driven ETF strategy backtest."""
from __future__ import annotations

import json

from backtest_module import run_backtest


if __name__ == "__main__":
    print(json.dumps(run_backtest(), indent=2))
