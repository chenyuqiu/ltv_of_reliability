# Delivery Marketplace Simulator — For Paper Reproduction

## Structure

```
ltv_of_reliability/
├── sim/                              # Core simulator
│   ├── params.py                     # DGPParams: ~50 tunable parameters
│   ├── users.py                      # User generative model + lifecycle Markov chain
│   ├── demand.py                     # Daily order count + per-order characteristics
│   └── marketplace.py                # 5-min slot FIFO courier matching + delay logic
├── ab_validation/
│   ├── run_simulation.py             # AB simulation + estimators → cache + Table 1
│   └── plot_figure.py                # Reads cache → figures
├── switchback_validation/
│   ├── run_simulation.py             # Switchback sim, 20 eater-resampled runs → Figure 5
│   ├── run_table_simulation.py       # 200 independent markets → Table 2 (printed to log)
│   └── plot_figure.py                # Reads results → Figure 5
└── run_all.sh                        # Full pipeline from scratch
```

## Reproduction

Run the full pipeline from scratch (simulations → figures + tables):

```bash
cd ltv_of_reliability
bash run_all.sh
```

## Dependencies

```
python >= 3.9
numpy
pandas
scipy
scikit-learn
matplotlib
```
