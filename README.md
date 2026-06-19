# MoQ Timeout Experiment Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20761155.svg)](https://doi.org/10.5281/zenodo.20761155)

Experiment data for evaluating RESET_AT vs RESET_STREAM timeout strategies in Media over QUIC (MoQ) live video streaming.

A citable snapshot is archived on Zenodo at [10.5281/zenodo.20761155](https://doi.org/10.5281/zenodo.20761155) (concept DOI [10.5281/zenodo.20761154](https://doi.org/10.5281/zenodo.20761154) always resolves to the latest version).

## Structure

```
data/                     # Raw experiment data (git-annex, needs datalad get)
├── baseline/             # No impairment
├── lossy/                # 2% packet loss, unlimited bandwidth
├── congested/            # 1 Mbit/s bandwidth cap
└── severe/               # 2% loss + 1 Mbit/s cap
    └── <strategy>-<timeout>/
        └── run<N>/       # 10 repetitions per configuration
            ├── relay_metrics.csv
            ├── sub1_display.csv
            ├── sub1_recv.csv
            ├── manifest.csv
            └── ...
analysis/                 # Analysis scripts
data/*.csv                # Summary CSVs (plain git, immediately available)
matrix.json               # Experiment parameters
```

## Quick Start

```bash
# Clone (summary CSVs are available immediately)
datalad clone https://github.com/harlequix/moq-timeout-dataset.git

# Regenerate summary CSVs from raw data
datalad get data/       # Fetch annexed raw data (~43 MB)
uv sync                 # Install Python dependencies
make                    # Run analysis pipeline
```

## Summary Files

| File | Description |
|---|---|
| `summary.csv` | Per-run delivery counts and relay metrics |
| `latency.csv` | Per-object end-to-end latency |
| `decodability.csv` | Decodable video duration and GoP-level stall analysis |
| `stalls.csv` | Per-frame stall events (inter-frame interval > 100ms) |
| `stall_summary.csv` | Per-run aggregated stall metrics |
| `summary_table.txt` | Human-readable pivot table |

All summary files were generated via `datalad run` and can be reproduced with `make`.

## Experiment Runner

These runs were produced by
[moq-timeout-experiments](https://github.com/harlequix/moq-timeout-experiments),
which drives the publisher, relay, and subscriber from
[moq-streaming](https://github.com/harlequix/moq-streaming).

## License

This dataset is released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
