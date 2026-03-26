from __future__ import annotations

import argparse
import json

from main import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="AEA agent runner")
    parser.add_argument(
        "question",
        nargs="?",
        default="Warfarin ile greyfurt suyu etkileşimi nedir?",
        help="User question (interaction / side effect / general info)",
    )
    args = parser.parse_args()

    result = run_agent(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

