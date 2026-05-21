# Lab 3: FEM Soft Body Simulation

## Run

From repository root:

```bash
uv run lab3 --demo basic
uv run lab3 --demo b1
uv run lab3 --demo b2
uv run lab3 --demo b3
uv run lab3 --demo b4
```

With debug logs:

```bash
uv run lab3 --demo basic --debug
```

Safe boot (to avoid GPU driver hangs / machine freeze):

```bash
uv run lab3 --demo basic --safe-boot --debug
```

`--safe-boot` mode:
- forces CPU backend,
- keeps only Low mesh preset,
- clamps substeps to conservative values.

Logs are written in real time to timestamped files, with `runtime.log` kept as the latest-run mirror:

```text
lab3/logs/runtime_YYYYMMDD-HHMMSS.log
lab3/logs/runtime.log
```

Equivalent command:

```bash
uv run python -m lab3 --demo basic
uv run python -m lab3 --demo b1
uv run python -m lab3 --demo b2
uv run python -m lab3 --demo b3
uv run python -m lab3 --demo b4
```
