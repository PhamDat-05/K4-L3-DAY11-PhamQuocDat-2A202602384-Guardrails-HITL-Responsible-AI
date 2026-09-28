"""Replay only CP4 rows that failed due to a transient provider error.

Successful rows remain the original observed responses. The combined and
per-target files are regenerated with the normal attack artifact writers.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.agent import create_red_agent_default
from agents.guards_agent import create_red_agent_advance
from attacks.attacks import run_attacks, save_attack_results, write_run_attack_json


async def main() -> int:
    path = ROOT / "outputs" / "attack_results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    targets = (
        ("unsafe_attacks", "red_default", create_red_agent_default),
        ("guards_attacks", "red_advance", create_red_agent_advance),
    )
    updated = False
    for key, target, factory in targets:
        rows = data[key]
        failed = [row for row in rows if row.get("layer") == "error"]
        if not failed:
            continue
        agent, runner = factory()
        prompts = [{"id": row["id"], "category": row["category"],
                    "input": row["input"]} for row in failed]
        replayed = await run_attacks(agent, runner, prompts=prompts,
                                     target_name=target, save_json=False)
        by_id = {row["id"]: row for row in replayed if row.get("layer") != "error"}
        if by_id:
            data[key] = [by_id.get(row["id"], row) for row in rows]
            write_run_attack_json(data[key], target_name=target)
            updated = True
    if updated:
        save_attack_results(unsafe_results=data["unsafe_attacks"],
                            guards_results=data["guards_attacks"])
    remaining = sum(row.get("layer") == "error"
                    for key in ("unsafe_attacks", "guards_attacks") for row in data[key])
    print(f"Remaining transient errors: {remaining}")
    return 1 if remaining else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
