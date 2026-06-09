#!/usr/bin/env bash
# run_all.sh — Full end-to-end pipeline from scratch.
#
# Runtime estimates:
#   AB simulation:                ~30–60 min
#   Switchback box-plot sim:      ~30–60 min  (20 forks × 3 arms)
#   Switchback Table-2 sim:       ~2–3 hours  (200 independent markets × 3 arms)
#   Plots / table:                seconds
#
# The pre-computed caches in ab_validation/cache/ and switchback_validation/results/
# (sw_validation_results.json, table2_results.json) will be overwritten.
set -e
cd "$(dirname "$0")"

PYTHON=${PYTHON:-/opt/piperenv/bin/python3}

echo "=== Step 1/5: AB Simulation ==="
$PYTHON ab_validation/run_simulation.py

echo ""
echo "=== Step 2/5: AB Figure ==="
$PYTHON ab_validation/plot_figure.py

echo ""
echo "=== Step 3/5: Switchback box-plot simulation (Fig 5) ==="
$PYTHON switchback_validation/run_simulation.py

echo ""
echo "=== Step 4/5: Switchback box-plot figure (Fig 5) ==="
$PYTHON switchback_validation/plot_figure.py

echo ""
echo "=== Step 5/5: Switchback Table-2 simulation (200 independent markets) ==="
# Reuses β from Step 3; prints Table 2 (bias / RMSE / coverage) to the log.
$PYTHON switchback_validation/run_table_simulation.py

echo ""
echo "Done. Figures written to figures/"
ls -lh figures/

