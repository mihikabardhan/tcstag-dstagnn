# Notebooks

| File | What it does | Status |
|---|---|---|
| `01_baseline_pems04.ipynb` | Retrains DSTAGNN on PEMS04 (3 seeds); writes `runner.py` | included |
| `02_tcstag_main.ipynb` | Trains `base`, `te`, `tcstag` (and ablation variants) with the shared runner; results tables; gate figure | included |
| regime graphs (PEMS04/PEMS08) | Builds the global and regime STAD graphs with POT | **TODO: copy `jinay_regime_graphs_pems08.ipynb` here** |
| alignment and noise checks | Calendar-alignment check and graph-noise control | **TODO: copy `checks_alignment_and_noise.ipynb` here** |
| conformal intervals | Time-aware conformal calibration study (variants A to F) | **TODO: copy `conformal_time_aware_pems04.ipynb` here** |

Notebooks are committed without outputs to keep the repository small. The three TODO notebooks hold results that appear in the paper (regime graphs, calendar checks, and the interval study), so add them before sharing the repository.
