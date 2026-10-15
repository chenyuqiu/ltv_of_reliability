# Delivery Marketplace Simulator — For Paper Reproduction

## Structure

```
ltv_of_reliability/
├── sim/                              # Core simulator
│   ├── params.py                     # DGPParams: ~50 tunable parameters
│   ├── users.py                      # User generative model + lifecycle Markov chain
│   ├── demand.py                     # Daily order count + per-order characteristics
│   ├── marketplace.py                # 5-min slot FIFO courier matching + delay logic
│   └── confounder.py                 # Optional unobserved quality-shock sensitivity
├── ab_validation/
│   ├── run_simulation.py             # Main AB simulation and Table 1
│   ├── diagnostics.py                # Balance, overlap, and weight diagnostics
│   └── plot_figure.py                # Figures 1–6 and 8
├── switchback_validation/
│   ├── run_simulation.py             # 20 evaluation forks → Figure 7
│   ├── run_table_simulation.py       # 200 evaluation forks → Table 2
│   ├── carryover_sensitivity/
│   │   └── run_different_burn_in.py  # 60/80-minute burn-in check
│   └── plot_figure.py                # Figure 7
├── sensitivity/
│   └── run_unobserved_confounder.py  # Unobserved-U scenarios → Table 7
├── figures/                          # Paper Figures 1–8 (PNG)
└── run_all.sh                        # End-to-end pipeline
```

## Reproduction

Run the pipeline from clean output directories:

```bash
cd ltv_of_reliability
bash run_all.sh
```

Figures are saved in `figures/`; each simulation prints its table results.

## Dependencies

```
python >= 3.9
numpy
pandas
scipy
scikit-learn
matplotlib
```

## Citation
```
Qiu, Chenyu, Xu Kuang, Inessa Liskovich, Ali Rauh, and Stefan Wager. 2026. "What is the Long-Term Value of Reliability?" arXiv preprint arXiv:2606.11526. https://arxiv.org/abs/2606.11526.
```
