from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_LOGS = {
    "StVK": "b1stvk.log",
    "Neo-Hookean": "b1neo.log",
    "Corotated": "b1cor.log",
}

PLOTS = {
    "total_energy": ("energy_total.png", "Total Energy", "total_energy"),
    "kinetic_energy": ("energy_kinetic.png", "Kinetic Energy", "kinetic_energy"),
    "potential_energy": ("energy_potential.png", "Potential Energy", "potential_energy"),
    # Current B1 logs record elastic energy as potential_elastic. Use it as the
    # compression/strain potential curve for model comparison.
    "potential_elastic": ("energy_compression.png", "Compression Potential Energy", "potential_elastic"),
}

FIELD_RE = re.compile(r"([A-Za-z_]+)=([^ ]+)")


def _parse_float(value: str) -> float | None:
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_analysis_log(path: Path, max_time: float) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "ANALYSIS" not in line:
                continue
            fields: dict[str, float | str] = {}
            for key, raw_value in FIELD_RE.findall(line):
                value = _parse_float(raw_value)
                fields[key] = raw_value if value is None else value
            t = fields.get("t")
            if not isinstance(t, float) or t > max_time:
                continue
            rows.append(fields)
    return rows


def plot_field(
    series: dict[str, list[dict[str, float | str]]],
    field: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)

    for model_name, rows in series.items():
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            t = row.get("t")
            value = row.get(field)
            if isinstance(t, float) and isinstance(value, float):
                xs.append(t)
                ys.append(value)
        if xs:
            ax.plot(xs, ys, linewidth=1.6, label=model_name)

    ax.set_title(f"B1 {title} (0-30s)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("E")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot B1 energy curves from demo logs.")
    parser.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parent / "logs")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "plots")
    parser.add_argument("--max-time", type=float, default=30.0)
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    series: dict[str, list[dict[str, float | str]]] = {}
    for model_name, filename in DEFAULT_LOGS.items():
        path = args.log_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing log file: {path}")
        rows = parse_analysis_log(path, args.max_time)
        if not rows:
            raise ValueError(f"no ANALYSIS rows found in {path} within 0-{args.max_time:g}s")
        series[model_name] = rows

    for field, (filename, title, _) in PLOTS.items():
        output_name = str(Path(filename).with_suffix(f".{args.format}"))
        plot_field(series, field, title, args.out_dir / output_name)

    print(f"Wrote {len(PLOTS)} plots to {args.out_dir}")


if __name__ == "__main__":
    main()
