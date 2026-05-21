from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


FIELD_RE = re.compile(r"([A-Za-z_]+)=([^ ]+)")


def _parse_float(value: str) -> float | None:
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_analysis(path: Path, max_time: float) -> tuple[list[float], list[float]]:
    ts: list[float] = []
    es: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "ANALYSIS" not in line:
                continue
            fields = {k: v for k, v in FIELD_RE.findall(line)}
            t = _parse_float(fields.get("t", ""))
            e = _parse_float(fields.get("total_energy", ""))
            if t is None or e is None:
                continue
            if t > max_time:
                continue
            ts.append(t)
            es.append(e)
    return ts, es


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot B4 total energy from low/high iteration logs.")
    parser.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parent / "logs")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "plots")
    parser.add_argument("--max-time", type=float, default=30.0)
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    args = parser.parse_args()

    low_log = args.log_dir / "b4low.log"
    high_log = args.log_dir / "b4high.log"
    if not low_log.exists() or not high_log.exists():
        raise FileNotFoundError(f"missing required logs: {low_log} / {high_log}")

    t_low, e_low = parse_analysis(low_log, args.max_time)
    t_high, e_high = parse_analysis(high_log, args.max_time)
    if not t_low or not t_high:
        raise ValueError("no valid ANALYSIS total_energy samples found in b4 logs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"b4_total_energy.{args.format}"

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)
    ax.plot(t_low, e_low, linewidth=1.6, label="Low Iter Steps")
    ax.plot(t_high, e_high, linewidth=1.6, label="High Iter Steps")
    ax.set_title("B4 Total Energy (0-30s)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("E")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
