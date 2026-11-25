# Burn or pay? Revenue distribution in a staking economy — replication package

Code, data and output files for the paper "Burn or pay? Revenue distribution in
a staking economy". Every number in the paper is taken from a file under an
`output/` folder in this repository or from the data files listed below.

## Requirements

Python 3.12 with the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Run each script from the folder it sits in, for example:

```bash
cd revision && python check_v5_stationary.py
```

Scripts write to the `output/` folder next to them. The lifecycle runs use
`multiprocessing`; the two 1,600-draw runs take several hours on eight cores,
the rest take seconds to minutes.

## Layout

The folder names follow the order in which the work was done. Table and item
numbers inside script docstrings refer to earlier drafts; use the map below.

| Folder | Contents |
|---|---|
| `gonogo/` | Core solver `cht.py` (pricing equation, staking block, burn share), first policy sweeps, simulated regression panels |
| `uq/` | Stationary hurdle under parameter uncertainty (20,000 draws) |
| `gate2/` | Two agent types (stakers and holders) and fee incidence |
| `revision/` | Lifecycle solver with centred feedback (`centred.py`, `centred2.py`), all lifecycle tables, Proposition 2, robustness runs, fee and staking data |
| `empirical/` | Governance votes: news windows, doses, abnormal returns, power |
| `draft/figs/` | `make_figs.py`, which draws every figure from the output files |

## Where each result comes from

| Result in the paper | Script | Output |
|---|---|---|
| Stationary hurdle, closed form checks | `revision/check_v5_stationary.py` | `revision/output/check_v5_stationary.txt` |
| Stationary hurdle under priors; quantiles of the hurdle over the reward rate | `uq/uq0.py`, `uq/uq0b.py` | `uq/output/uq0*.txt`, `uq0_draws.parquet` |
| Sobol indices | `revision/sobol_ratio.py` | `revision/output/sobol_ratio.txt` |
| Liquid staking | `revision/liquid.py` | `revision/output/liquid.csv` |
| Lifecycle gaps and hurdles (main table) | `revision/riskprem2.py`, `revision/table_c.py`, `revision/check_v5_domain.py` | `revision/output/riskprem2.csv`, `table_centred.csv`, `check_v5_domain.txt` |
| Ratio of hurdles by fee yield | `revision/check_v5_feelevel12.py` | `revision/output/check_v5_feelevel12.txt` |
| Proposition 2: weights, local hurdles, covariance | `revision/prop2.py`, `revision/check_v5_qsign.py` | `revision/output/prop2_check.csv`, `prop2_weights.csv` |
| Fixed-centre feedback | `revision/check_v5_chtform.py` | `revision/output/check_v5_chtform.txt` |
| Baseline burn share and fee floor | `revision/check_v6_baseline.py`, `revision/check_v6_omega09_rf10.py` | `revision/output/check_v6_*` |
| Lifecycle hurdles under priors (1,600 draws) | `revision/uq3.py`, `revision/uq3_retry.py`, `revision/uq3_report.py` | `revision/output/uq3_*` |
| Risk premium in the drift | `revision/riskprem2.py`, `revision/uq4.py`, `revision/uq4_report.py` | `revision/output/riskprem2.*`, `uq4_*` |
| Fee speed and the fee elasticity | `revision/feeelast.py`, `revision/feespeed.py` | `revision/output/feeelast.csv`, `feespeed.csv` |
| Numerical uniqueness | `revision/uniq.py` | `revision/output/uniq.txt` |
| Stakers and holders; fees paid by holders | `gate2/hetero.py`, `gate2/sweep2.py` | `gate2/output/` |
| Simulated predictive regressions | `gonogo/panel.py`, `gonogo/panel_highk.py` | `gonogo/output/panel*.txt` |
| Fee revenue elasticity (data) | `revision/feedata/fetch.py`, `elasticity.py`, `fee_sim.py` | `revision/feedata/*.csv` |
| Staking response (data) | `revision/stakedata/fetch_staking*.py`, `staking_response.py` | `revision/stakedata/*.csv`, `staking_response.txt` |
| Governance votes: returns and power | `empirical/e2_returns.py`, `empirical/e2_power.py`, `revision/votes_nodydx.py` | `empirical/output/e2_*`, `revision/output/votes_nodydx.txt` |
| Figures | `draft/figs/make_figs.py` | PDF files next to the script |

Folders named `redteam*` hold adversarial checks run during review; they are
not needed to reproduce the paper.

## Data

All data were retrieved on 9–10 November 2026 and are stored here as
retrieved; the `fetch*.py` scripts show the calls but will return later
vintages if re-run.

| Data | Source | Files |
|---|---|---|
| Daily chain fee revenue | DefiLlama chain-fee series | `revision/feedata/fees_mcap_daily.csv` |
| Market capitalisation | Coin Metrics community data, `CapMrktEstUSD` (CC BY-NC 4.0) | `revision/feedata/fees_mcap_daily.csv` |
| Staking ratios, reward and issuance rates | Public chain endpoints and explorers | `revision/stakedata/raw/`, `staking_panel.csv` |
| Daily and hourly token prices | Binance public klines | `empirical/output/hourly_cache/`, `empirical/output/e2_returns.csv` |
| Vote windows and doses | Governance forums and on-chain proposals, coded by hand | `empirical/e2_windows.csv`, `empirical/*dose*.csv` |

Third-party data remain subject to their providers' terms. Coin Metrics
community data are redistributed under CC BY-NC 4.0 with attribution.

## Licence

Code: MIT (see `LICENSE`). Third-party data: see above.
