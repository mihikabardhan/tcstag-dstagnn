# TC-STAG: time-aware forecasting and calibration for DSTAGNN

Code for the paper **"Time-Aware Forecasting and Calibration for DSTAGNN: What Helps, What Does Not, and Why"**
(Bardhan, Velkumar, Hiran, Anand; Manipal Institute of Technology).

We study whether making DSTAGNN's graph time-dependent helps traffic forecasting on PEMS04 (and PEMS08), and we test time-aware conformal prediction intervals.

## Main findings (PEMS04, 3 seeds)

| Model | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| DSTAGNN (paper) | 19.30 | 31.46 | 12.70 |
| DSTAGNN (ours) | 19.344 ± 0.083 | 31.603 ± 0.179 | 12.817 ± 0.055 |
| + time embeddings (`te`) | 19.167 ± 0.053 | 31.311 ± 0.149 | 12.668 ± 0.083 |
| TC-STAG (`tcstag`) | 19.201 ± 0.008 | 31.423 ± 0.038 | 12.696 ± 0.023 |

- Time embeddings lower MAE by about 0.9%, in every regime and at every horizon.
- The time-conditioned graph mixture (TC-STAG) gives no distinguishable gain over time embeddings and trains about 31% slower.
- Time-aware conformal calibration (time-of-day cells or forecast-level scaling) lowers the interval score by 8-10% against a single global calibration.
- A control run (baseline retrained in our runner, seed 2: MAE 19.385 vs 19.440 from the original script) shows the pipeline difference is small, but it is a single seed.

Per-seed numbers are in `results/pems04_per_seed.csv`. Differences below about 0.1 MAE are within seed noise.

## Repository layout

```
src/            model and training scripts (tcstag.py, runner_tc.py, runner_baseline.py)
notebooks/      Kaggle notebooks that run the experiments (see notebooks/README.md)
analysis/       analyze_tc.py (pooled stats, bootstrap, ensembles), make_gate_figure.py
results/        per-seed results
paper/          LaTeX source of the paper (figures to be added)
scripts/        setup_dstagnn.sh (clones the authors' code and installs our model file)
```

## Reproducing

We build on the authors' code (<https://github.com/SYLan2019/DSTAGNN>), which we do not redistribute.

1. `pip install -r requirements.txt`
2. `bash scripts/setup_dstagnn.sh` clones the authors' repo into `third_party/` and copies `src/tcstag.py` into its `model/` folder.
3. Put `PEMS04.npz` and `PEMS04.csv` into `third_party/DSTAGNN/data/PEMS04/` (PEMS data is not included here) and run the authors' `prepareData.py` as in notebook 01.
4. Build the regime graphs (global, weekday peak, weekday off-peak, weekend) with POT (notebook for regime graphs, see `notebooks/README.md`).
5. Train one run, for example:
   ```
   python src/runner_tc.py --repo third_party/DSTAGNN --dataset PEMS04 --variant tcstag \
       --seed 1 --max_epochs 60 --patience 15 --graph AG --start_dow 0 \
       --graph_dir <regime graph folder> --out runs/PEMS04_tcstag_seed1
   ```
   Variants: `base`, `te`, `tcstag` (full runs); `tcstag_prior`, `hard`, `soft_static` are implemented but were not trained to completion. The baseline script is `src/runner_baseline.py`.
6. Pool the runs and compute the paper's statistics:
   ```
   python analysis/analyze_tc.py runs/
   python analysis/make_gate_figure.py runs/
   ```
   Run folders must be named `PEMS04_<variant>_seed<k>`, with baseline runs named `PEMS04_AG_seed<k>`.

Protocol: Adam, learning rate 1e-4, batch 32, Huber loss, at most 60 epochs with patience 15, best-validation checkpoint tested. PEMS04 starts on Monday 1 Jan 2018 (`--start_dow 0`); PEMS08 starts on Friday (`--start_dow 4`).

## Notes and limitations

- Three seeds per model; the test period is about 12 days.
- The PEMS08 baseline may have used a different graph sparsity than the final setting (0.02); see the paper.
- Large artefacts (prediction files, checkpoints, data) are not in this repository.

## License and citation

Our code is released under the MIT license (see `LICENSE`). The authors' DSTAGNN code has its own license; please check it. Citation entry: TODO after the paper is public.
