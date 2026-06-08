"""Export utilities for IMDB product datasets."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Sequence

import pandas as pd

from .models import ProductRecord


EXPORT_DIR = Path("exports")


class Exporter:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or EXPORT_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def export(self, records: Sequence[ProductRecord], format: str) -> Path:
        normalized_format = format.lower()
        if normalized_format not in {"csv", "excel"}:
            msg = f"Unsupported export format: {format}"
            raise ValueError(msg)

        timestamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        path = self.base_dir / f"imdb-export-{timestamp}.{self._extension(normalized_format)}"

        df = pd.DataFrame([record.values_for_export() for record in records])

        if normalized_format == "csv":
            df.to_csv(path, index=False)
        else:
            df.to_excel(path, index=False)

        return path

    @staticmethod
    def _extension(format: str) -> str:
        return "csv" if format == "csv" else "xlsx"

