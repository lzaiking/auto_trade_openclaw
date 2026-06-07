"""Validate point-in-time monthly factor inputs before model training."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from typing import Dict, List, Optional

from market_data import FACTOR_PATH, REPORT_DIR
from prediction_module import FACTOR_COLUMNS, FACTOR_FEATURES, ensure_factor_template, parse_optional_float


FACTOR_SCHEMA = {
    "eps_revision_score": {
        "description": "Monthly EPS revision strength. Higher is more bullish.",
        "min": -5.0,
        "max": 5.0,
        "source_note": "Use point-in-time analyst estimate data available before the monthly decision date.",
    },
    "forward_peg_score": {
        "description": "Forward PEG attractiveness score. Higher is more bullish.",
        "min": -5.0,
        "max": 5.0,
        "source_note": "Normalize from point-in-time valuation data available before the monthly decision date.",
    },
    "fcf_yield_minus_tbill": {
        "description": "Free-cash-flow yield minus short Treasury yield, expressed as a decimal spread.",
        "min": -1.0,
        "max": 1.0,
        "source_note": "Use point-in-time index FCF yield and short Treasury yield available before the monthly decision date.",
    },
    "ai_profit_conversion_score": {
        "description": "AI capex to profit conversion score. Higher is more bullish.",
        "min": -5.0,
        "max": 5.0,
        "source_note": "Use data available before the monthly decision date.",
    },
    "top_weight_concentration": {
        "description": "Top constituent weight concentration, decimal 0-1.",
        "min": 0.0,
        "max": 1.0,
        "source_note": "Use index composition available before the monthly decision date.",
    },
    "breadth_200dma": {
        "description": "Share of index constituents above 200-day moving average, decimal 0-1.",
        "min": 0.0,
        "max": 1.0,
        "source_note": "Use constituent prices available before the monthly decision date.",
    },
    "crowding_score": {
        "description": "Crowding/positioning heat score. Higher means more crowded unless normalized otherwise.",
        "min": -5.0,
        "max": 5.0,
        "source_note": "Use point-in-time positioning or flow data available before the monthly decision date.",
    },
}


def _empty_feature_stats() -> Dict[str, Dict[str, object]]:
    """生成空的因子统计结构；当文件无数据时也能输出完整验证报告。"""

    return {
        name: {
            "present_count": 0,
            "coverage": 0.0,
            "min": None,
            "max": None,
            "out_of_range_count": 0,
        }
        for name in FACTOR_FEATURES
    }


def validate_factor_file() -> Dict[str, object]:
    """验证月度因子文件结构、日期、重复月份、数值范围和覆盖率，并写出 JSON 报告。"""

    ensure_factor_template()
    errors: List[str] = []
    warnings: List[str] = []
    rows = []
    feature_values: Dict[str, List[float]] = {name: [] for name in FACTOR_FEATURES}
    duplicate_months = set()
    seen_months = set()

    with FACTOR_PATH.open() as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        missing_columns = [name for name in FACTOR_COLUMNS if name not in columns]
        extra_columns = [name for name in columns if name not in FACTOR_COLUMNS]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        if extra_columns:
            warnings.append(f"Extra columns ignored by the model: {', '.join(extra_columns)}")

        for line_no, row in enumerate(reader, start=2):
            raw_date = (row.get("date") or "").strip()
            try:
                d = datetime.strptime(raw_date, "%Y-%m-%d").date()
            except ValueError:
                errors.append(f"Line {line_no}: invalid date {raw_date!r}; expected YYYY-MM-DD")
                continue

            month_key = (d.year, d.month)
            if month_key in seen_months:
                duplicate_months.add(f"{d.year:04d}-{d.month:02d}")
            seen_months.add(month_key)
            rows.append(row)

            for name in FACTOR_FEATURES:
                value = parse_optional_float(row.get(name))
                if value is None:
                    continue
                feature_values[name].append(value)
                schema = FACTOR_SCHEMA[name]
                if value < schema["min"] or value > schema["max"]:
                    errors.append(
                        f"Line {line_no}: {name}={value} outside [{schema['min']}, {schema['max']}]"
                    )

    if duplicate_months:
        errors.append(f"Duplicate factor months: {', '.join(sorted(duplicate_months))}")

    row_count = len(rows)
    feature_stats = _empty_feature_stats()
    for name, values in feature_values.items():
        schema = FACTOR_SCHEMA[name]
        out_of_range = [value for value in values if value < schema["min"] or value > schema["max"]]
        feature_stats[name] = {
            "present_count": len(values),
            "coverage": round(len(values) / row_count, 4) if row_count else 0.0,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "out_of_range_count": len(out_of_range),
        }

    if row_count == 0:
        warnings.append("monthly_factors.csv has only a header; fundamental factor coverage is 0%.")
    else:
        low_coverage = [
            name for name, stats in feature_stats.items()
            if stats["coverage"] < 0.8
        ]
        if low_coverage:
            warnings.append(f"Low factor coverage below 80%: {', '.join(low_coverage)}")

    result = {
        "path": str(FACTOR_PATH),
        "passed_structure": not errors,
        "row_count": row_count,
        "required_columns": FACTOR_COLUMNS,
        "schema": FACTOR_SCHEMA,
        "feature_stats": feature_stats,
        "errors": errors,
        "warnings": warnings,
        "point_in_time_rule": (
            "Each row must contain only data observable before that month's strategy decision date. "
            "Do not revise historical rows with later analyst estimates, index composition, or positioning data."
        ),
    }
    REPORT_DIR.mkdir(exist_ok=True)
    with (REPORT_DIR / "factor_validation.json").open("w") as f:
        json.dump(result, f, indent=2)
    return result
