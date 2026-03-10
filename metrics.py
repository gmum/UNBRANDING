#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

ALPHA = 0.5
BETA = 0.5
MAX_SIM = 10.0
MAP = {
    "adidas": "ADIDAS",
    "apple": "APPLE",
    "audi": "AUDI",
    "bmw": "BMW",
    "coca-cola": "COCA_COLA",
    "emirates": "EMIRATES",
    "mcdonalds": "MCDONALDS",
    "mercedes": "MERCEDES",
    "monster": "MONSTER",
    "nike": "NIKE",
    "puma": "PUMA",
    "singapore": "SINGAPORE_AIRLINES",
}


def first_jsonl(path: Path) -> Path:
    if path.is_file():
        return path
    files = sorted(path.glob("eval_*.jsonl"))
    if not files:
        raise SystemExit(f"No eval_*.jsonl in {path}")
    return files[0]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def expected_label(filename: str) -> str:
    prefix = Path(filename).name.split("_", 1)[0].lower()
    return MAP.get(prefix, prefix.upper())


def main() -> None:
    p = argparse.ArgumentParser(description="Compute B/S/U from BPS+VSS JSONL.")
    p.add_argument("--bps", type=Path, default=Path("results/example/bps"))
    p.add_argument("--vss", type=Path, default=Path("results/example/vss"))
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--beta", type=float, default=BETA)
    args = p.parse_args()

    bps = read_jsonl(first_jsonl(args.bps))
    vss = read_jsonl(first_jsonl(args.vss))

    unbrand = [r for r in bps if "unbrand" in str(r.get("filename", "")).lower()]
    if not unbrand:
        raise SystemExit("No unbrand rows in BPS file.")

    correct = sum(
        str(r.get("predicted_brand", "")).strip().upper() == expected_label(str(r.get("filename", "")))
        for r in unbrand
    )
    b = correct / len(unbrand)

    vss_map = {
        Path(str(r.get("filename", ""))).name: float(r["similarity_score"])
        for r in vss
        if isinstance(r.get("similarity_score"), (int, float))
    }
    sims = [vss_map[Path(str(r.get("filename", ""))).name] for r in unbrand if Path(str(r.get("filename", ""))).name in vss_map]
    if not sims:
        raise SystemExit("No overlapping similarity scores between BPS and VSS.")
    s = (sum(sims) / len(sims)) / MAX_SIM

    u = args.alpha * s + args.beta * (1.0 - b)

    print(f"B={b:.6f} ({b*100:.2f}%)")
    print(f"S={s:.6f} ({s*100:.2f}%)")
    print(f"U={u:.6f} ({u*100:.2f}%)")


if __name__ == "__main__":
    main()
