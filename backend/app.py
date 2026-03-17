from __future__ import annotations

import json

from backend.graph.flow import run_once


def main() -> None:
    # Minimal CLI smoke test (non-interactive by default).
    result = run_once("warfarin", "greyfurt suyu")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

