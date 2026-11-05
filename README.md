# SAelection — Gauteng 2026 live VD / ward projection

Bayesian spatial model that predicts the **2026 Local Government Election (PR ballot) in Gauteng**
for every **voting district (VD)** and every **ward**: votes for the DA, ANC, ActionSA, MK, EFF
and Other, plus turnout. During counting it takes the VDs already counted and projects the votes
still to come and the final distribution, without being fooled by the order in which VDs report.

* 2,797 VDs, 526 wards, 9 municipalities (2026 delimitation)
* trained on VD-level IEC results for 2016 LGE, 2019 PE, 2021 LGE and 2024 PE
* fitted with variational inference (NumPyro SVI); runs on a laptop CPU or on Kaggle

## The model

For VD *i* and party *k* (log-ratio of the party's votes to non-voters, so one model covers both
party choice and turnout):

```
eta_ik = A_k                         provincial swing level             (from the count / polls)
       + delta_m(i),k                municipal swing                    (from the count)
       + b1_k X_LGE2021,ik           own past performance, last LGE     (learned 2016->2021)
       + b2_k X_PE2024,ik            own past performance, last PE
       + sum_j C_kj X_j              SUBSTITUTION: past strength of ideologically similar
                                     parties j (config/ideology.json: DA~ActionSA, ANC~MK~EFF,
                                     DA/ActionSA~Other)
       + phi_ik + g_i                SPATIAL: Leroux CAR over VDs whose polygons touch,
                                     one field per party + a common turnout field
       + psi_ik                      VD residual, correlated across parties (LKJ prior);
                                     negative correlations = substitution inside a VD
votes_i ~ Multinomial(registered_i, softmax(eta_i, 0))
```

**Spatial term.** Only contiguity matters: VDs are neighbours iff their boundaries touch.
For each party, λ_k measures how strongly touching VDs co-vary. A hold-out test (20% of VDs) compares
the model with and without the spatial term. The term is switched off for any party where it does not
improve out-of-sample prediction (`artifacts/spatial_test.json`).

**Live counting.** Counted VDs are fixed at their actual result. Each one is compared with *its
own* history, so the count tells the model about the swing (A, δ). Through the CAR term, it also
tells the model about the uncounted VDs that touch it. A party that leads early only because its
strongholds counted first is not projected to keep that lead. `artifacts/backtest_summary.csv`
replays the 2024 count in deliberately biased orders to check this.

## Running it

```bash
pip install -r requirements.txt
```
```bash
bash scripts/00_fetch_data.sh        # raw IEC results + boundaries (~1 GB, not committed)
```
```bash
python scripts/prepare_data.py       # data/processed/: VD history, contiguity graphs, VD->ward crosswalk
```
```bash
python scripts/run_train.py          # artifacts/: learned structure, spatial hold-out test (~10 min CPU)
```
```bash
PYTHONPATH=src python -m saelection.backtest   # artifacts/backtest_*.csv (count replay on 2024)
```
```bash
python scripts/run_live.py           # output/live/: projections for the frontend
```

**Live count (IEC dashboard feed).** `scripts/update_live.sh` runs one refresh cycle:
`python -m saelection.iec_live` pulls `results.elections.org.za/dashboards/lge/MapsJason/2026/EEID1887Munic<id>.js`
for the 9 Gauteng municipalities (per-VD votes, counting progress, 2026 registered voters). It converts the
per-VD all-ballot votes to PR-equivalent votes, re-runs the projection, and exports the web-app files to
`output/frontend/` (`predictions.json`, `results.json`, `wards.geojson`, in the contract of the map app on branch
`claude/eager-mayer-vzaen0`). About 3 minutes per cycle on 4 CPUs.

Alternatively, put the IEC 2026 per-VD result CSVs in `data/observed_2026/`
(see the README there) and re-run `scripts/run_live.py`. Before any VD is counted, the projection
uses the poll prior in `config/live.json`.

**Kaggle:** `kaggle/saelection_gauteng_2026.ipynb` runs the whole pipeline. Set your username in
`kaggle/kernel-metadata.json`, then run `kaggle kernels push -p kaggle`, or import the notebook
into Kaggle and turn Internet on.

## Output for the frontend (`output/live/`)

All files are regenerated on every run. JSON is the primary format and CSV is a flat copy of it.
Join to the maps by `id`.

| file | one record per | key fields |
|---|---|---|
| `meta.json` | run | `generated_at`, `n_vds`, `n_vds_counted`, `pct_vds_counted`, `registered_total`, `prior`, `model` (spatial λ per party, substitution matrices, residual correlations), `notes` |
| `province.json` | Gauteng | unit record (below) + `counted_so_far` (actual votes counted per category) |
| `municipalities.json/.csv` | municipality (9) | unit record + `name` |
| `wards.json/.csv` | ward (526) | unit record + `muni`, `ward_no`, `n_vd` |
| `vds.json/.csv` | VD (2,797) | unit record + `muni`, `ward` (main ward), `station`, `lon`, `lat`, `status` = `counted` or `projected` |
| `province_draws.csv` | posterior draw | final vote totals per category for each of the 500 draws (histograms / distributions) |
| `municipality_draws.csv` | draw × municipality | same, per municipality |
| `geo/vds_2026.geojson`, `geo/wards_2026.geojson` | polygon | `id` = VD number / ward id (simplified boundaries for mapping) |

**Unit record** (same shape at every level):

```json
{
  "id": 79800012,
  "pct_vds_counted": 37.5,
  "registered": 41234,
  "turnout_valid": {"mean": 0.41, "p05": 0.38, "p50": 0.41, "p95": 0.44},
  "parties": {
    "DA":  {"votes": {"mean": 6120, "p05": 5410, "p50": 6105, "p95": 6890},
            "share": {"mean": 0.36, "p05": 0.32, "p50": 0.36, "p95": 0.40},
            "p_first": 0.71},
    "ANC": {"...": "..."}, "ACTIONSA": {}, "MK": {}, "EFF": {}, "OTHER": {}
  },
  "projected_leader": "DA"
}
```

* `votes` = projected **final** PR votes, including votes already counted. For a counted VD,
  mean = p05 = p95 = the actual count.
* `share` = share of valid votes. `turnout_valid` = valid votes / registered voters.
* `p_first` = probability that the party finishes with the most votes in that unit.
* Remaining votes still to come = `votes` − `counted_so_far`, or per VD where `status` is `projected`.

## Data sources

* IEC VD-level results (results.elections.org.za), as republished in
  `xyclone-designs/database.iec`
* MDB/IEC voting-district and ward boundaries (2021, 2024, 2026), as mirrored in
  `garyvdm/demarcation-compare`
* Poll prior: Social Research Foundation (Q1 2026) and eNCA/MarkData Gauteng polls (see `config/live.json`)
