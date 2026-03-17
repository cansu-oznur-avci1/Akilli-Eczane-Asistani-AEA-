from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


class RiskLevel(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class InteractionRecord:
    ilac_adi: str
    etkilesen_madde: str
    risk_seviyesi: RiskLevel
    kaynak: str


def _norm(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def _default_csv_path() -> Path:
    env = os.getenv("AEA_RULE_TABLE_PATH")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "data" / "etkilesimler.csv"


class RuleEngine:
    """
    Deterministic risk lookup (no LLM decisions).

    - Reads a curated CSV rule table.
    - Returns one of: HIGH, LOW, NONE, UNKNOWN.
    """

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        self.csv_path = csv_path or _default_csv_path()
        self._index: Dict[Tuple[str, str], InteractionRecord] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Rule table not found at {self.csv_path}")

        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"ilac_adi", "etkilesen_madde", "risk_seviyesi", "kaynak"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"CSV headers must include {sorted(required)}; got {reader.fieldnames}"
                )

            for row in reader:
                ilac = (row.get("ilac_adi") or "").strip()
                madde = (row.get("etkilesen_madde") or "").strip()
                risk_raw = (row.get("risk_seviyesi") or "").strip().upper()
                kaynak = (row.get("kaynak") or "").strip()

                if not ilac or not madde or not risk_raw:
                    continue

                try:
                    risk = RiskLevel(risk_raw)
                except ValueError:
                    risk = RiskLevel.UNKNOWN

                rec = InteractionRecord(
                    ilac_adi=ilac,
                    etkilesen_madde=madde,
                    risk_seviyesi=risk,
                    kaynak=kaynak,
                )
                self._index[(_norm(ilac), _norm(madde))] = rec

        self._loaded = True

    def lookup(self, ilac_adi: str, etkilesen_madde: str) -> InteractionRecord | None:
        self.load()
        return self._index.get((_norm(ilac_adi), _norm(etkilesen_madde)))

    def risk_of(self, ilac_adi: str, etkilesen_madde: str) -> RiskLevel:
        rec = self.lookup(ilac_adi, etkilesen_madde)
        return rec.risk_seviyesi if rec else RiskLevel.UNKNOWN

    def iter_rules(self) -> Iterable[InteractionRecord]:
        self.load()
        return self._index.values()

