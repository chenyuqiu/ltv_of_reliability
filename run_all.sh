#!/usr/bin/env bash
# run_all.sh — Full end-to-end pipeline for the updated paper and response.
# Existing simulation caches and figures are overwritten.
set -e
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "=== Step 1: AB simulation (Table 1) ==="
"$PYTHON" ab_validation/run_simulation.py

echo ""
echo "=== Step 2: AB figures (Figures 1–6, 8) ==="
"$PYTHON" ab_validation/plot_figure.py

echo ""
echo "=== Step 3: Simulate switchback data (20 eater-path reruns) ==="
"$PYTHON" switchback_validation/run_simulation.py

echo ""
echo "=== Step 4: Plot Figure 7 from switchback data ==="
"$PYTHON" switchback_validation/plot_figure.py

echo ""
echo "=== Step 5: Switchback table (Table 2) ==="
"$PYTHON" switchback_validation/run_table_simulation.py

echo ""
echo "=== Step 6: Unobserved-confounder sensitivity (Table 7) ==="
for scenario in quality_delay_only quality_7d quality_14d; do
    REV_SCENARIO="$scenario" "$PYTHON" sensitivity/run_unobserved_confounder.py
done

echo ""
echo "Done. Figures written to figures/; tables printed above."
ls -lh figures/
