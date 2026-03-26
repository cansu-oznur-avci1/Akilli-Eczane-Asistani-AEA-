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


class QueryType(str, Enum):
    INTERACTION = "interaction"
    SIDE_EFFECT = "side_effect"
    GENERAL_INFO = "general_info"


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
        self._index_interaction: Dict[Tuple[str, str], InteractionRecord] = {}
        self._index_side_effect: Dict[Tuple[str, str], InteractionRecord] = {}
        self._index_general_info: Dict[Tuple[str, str], InteractionRecord] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Rule table not found at {self.csv_path}")

        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            headers = set(reader.fieldnames or [])

            interaction_required = {"ilac_adi", "etkilesen_madde", "risk_seviyesi", "kaynak"}
            side_effect_required = {"ilac_adi", "yan_etki", "risk_seviyesi", "kaynak"}
            general_info_required = {
                "ilac_adi",
                "genel_bilgi_konusu",
                "risk_seviyesi",
                "kaynak",
            }

            if not interaction_required.issubset(headers):
                raise ValueError(
                    "CSV headers must include interaction columns: "
                    f"{sorted(interaction_required)}; got {reader.fieldnames}"
                )

            for row in reader:
                ilac = (row.get("ilac_adi") or "").strip()
                kaynak = (row.get("kaynak") or "").strip()

                # 1) Interaction index (mandatory for the current CSV format)
                madde = (row.get("etkilesen_madde") or "").strip()
                risk_raw = (row.get("risk_seviyesi") or "").strip().upper()
                try:
                    risk = RiskLevel(risk_raw)
                except ValueError:
                    risk = RiskLevel.UNKNOWN

                if ilac and madde and risk_raw:
                    rec = InteractionRecord(
                        ilac_adi=ilac,
                        etkilesen_madde=madde,
                        risk_seviyesi=risk,
                        kaynak=kaynak,
                    )
                    self._index_interaction[(_norm(ilac), _norm(madde))] = rec

                # 2) Optional side-effect index (if CSV contains it)
                if side_effect_required.issubset(headers):
                    yan_etki = (row.get("yan_etki") or "").strip()
                    if ilac and yan_etki and risk_raw:
                        rec = InteractionRecord(
                            ilac_adi=ilac,
                            etkilesen_madde=yan_etki,
                            risk_seviyesi=risk,
                            kaynak=kaynak,
                        )
                        self._index_side_effect[(_norm(ilac), _norm(yan_etki))] = rec

                # 3) Optional general-info index (if CSV contains it)
                if general_info_required.issubset(headers):
                    genel_konu = (row.get("genel_bilgi_konusu") or "").strip()
                    if ilac and genel_konu and risk_raw:
                        rec = InteractionRecord(
                            ilac_adi=ilac,
                            etkilesen_madde=genel_konu,
                            risk_seviyesi=risk,
                            kaynak=kaynak,
                        )
                        self._index_general_info[(_norm(ilac), _norm(genel_konu))] = rec

        self._loaded = True

    def lookup_interaction(
        self, ilac_adi: str, etkilesen_madde: str
    ) -> InteractionRecord | None:
        self.load()
        drug = _norm(ilac_adi)
        target = _norm(etkilesen_madde)
        rec = self._index_interaction.get((drug, target))
        if rec:
            return rec

        # Fallback: tolerate minor phrasing differences in "etkilesen_madde"
        # by checking substring / token overlap.
        for (drug_key, madde_key), candidate in self._index_interaction.items():
            if drug_key != drug:
                continue
            if not madde_key or not target:
                continue

            if target in madde_key or madde_key in target:
                return candidate

            t_tokens = set(target.split())
            m_tokens = set(madde_key.split())
            overlap = t_tokens.intersection(m_tokens)
            if not overlap:
                continue

            # Require at least one reasonably long shared token
            # to avoid matching on generic short words.
            if any(len(tok) >= 4 for tok in overlap):
                return candidate

        return None

    def lookup_typed(
        self, ilac_adi: str, hedef: str, query_type: QueryType
    ) -> InteractionRecord | None:
        self.load()
        key = (_norm(ilac_adi), _norm(hedef))
        if query_type == QueryType.INTERACTION:
            return self.lookup_interaction(ilac_adi, hedef)
        if query_type == QueryType.SIDE_EFFECT:
            return self._index_side_effect.get(key)
        if query_type == QueryType.GENERAL_INFO:
            return self._index_general_info.get(key)
        return None

    def risk_of_typed(
        self, ilac_adi: str, hedef: str, query_type: QueryType
    ) -> RiskLevel:
        rec = self.lookup_typed(ilac_adi, hedef, query_type)
        return rec.risk_seviyesi if rec else RiskLevel.UNKNOWN

    # Backward-compatible API (interaction-only)
    def lookup(self, ilac_adi: str, etkilesen_madde: str) -> InteractionRecord | None:
        return self.lookup_interaction(ilac_adi, etkilesen_madde)

    def risk_of(self, ilac_adi: str, etkilesen_madde: str) -> RiskLevel:
        return self.risk_of_typed(ilac_adi, etkilesen_madde, QueryType.INTERACTION)

    def iter_rules(self) -> Iterable[InteractionRecord]:
        self.load()
        return list(self._index_interaction.values()) + list(self._index_side_effect.values()) + list(self._index_general_info.values())

