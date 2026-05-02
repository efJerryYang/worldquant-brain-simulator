from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import yaml


SAMPLE_RANGES: dict[str, tuple[date, date]] = {
    "test": (date(2019, 3, 1), date(2020, 9, 1)),
    "insample": (date(2015, 6, 1), date(2021, 3, 1)),
    "outsample": (date(2015, 6, 1), date(2022, 3, 1)),
    "latest": (date(2015, 6, 1), date(2023, 3, 1)),
}


@dataclass(frozen=True)
class SimulatorConfig:
    database_path: Path = Path("data/stock_snowball_us.db")
    table: str = "stock_data_US"
    sample: str = "test"
    universe: str = "Top3000"
    delay: int = 1
    neutralization: str = "Market"
    post_process_mode: str = "legacy"
    truncation: float = 0.01
    booksize: float = 20_000_000.0
    start_date: date | None = None
    end_date: date | None = None
    load_start_date: date | None = None
    output_dir: Path = Path("tmp")
    plot: bool = True
    min_symbols_per_day: int = 200

    @property
    def load_range(self) -> tuple[date, date]:
        fallback = SAMPLE_RANGES.get(self.sample.lower(), SAMPLE_RANGES["insample"])
        start = self.load_start_date or fallback[0]
        end = self.end_date or fallback[1]
        return start, end

    @property
    def simulation_start_date(self) -> date | None:
        if self.start_date is not None:
            return self.start_date
        if self.sample.lower() == "test":
            return date(2019, 11, 1)
        if self.sample.lower() == "insample":
            return date(2016, 3, 1)
        return None

    @property
    def universe_size(self) -> int:
        digits = "".join(ch for ch in self.universe if ch.isdigit())
        return int(digits) if digits else 3000


def parse_date(value: str | date | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(value)


def load_config(path: Path | str | None = None, **overrides: Any) -> SimulatorConfig:
    values: dict[str, Any] = {}
    if path is not None:
        config_path = Path(path)
        if config_path.exists():
            raw = yaml.safe_load(config_path.read_text()) or {}
            values.update(_normalize_yaml_keys(raw))

    for key in ("database_path", "output_dir"):
        if key in values:
            values[key] = Path(values[key])
    for key in ("start_date", "end_date", "load_start_date"):
        if key in values:
            values[key] = parse_date(values[key])

    config = SimulatorConfig(**values)
    clean_overrides = {
        key: value for key, value in overrides.items() if value is not None and hasattr(config, key)
    }
    for key in ("database_path", "output_dir"):
        if key in clean_overrides:
            clean_overrides[key] = Path(clean_overrides[key])
    for key in ("start_date", "end_date", "load_start_date"):
        if key in clean_overrides:
            clean_overrides[key] = parse_date(clean_overrides[key])
    return replace(config, **clean_overrides)


def _normalize_yaml_keys(raw: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "unit-handling": "unit_handling",
        "nan-handling": "nan_handling",
        "instrument-type": "instrument_type",
        "database": "database_path",
        "database-path": "database_path",
        "output-dir": "output_dir",
    }
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        config_key = aliases.get(key, key.replace("-", "_"))
        if config_key in SimulatorConfig.__dataclass_fields__:
            normalized[config_key] = value
    return normalized
