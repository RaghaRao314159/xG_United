# Tabular models (Ragha)

Gradient-boosted trees (LightGBM) and an in-context tabular foundation model (NVIDIA's Kumo
Tabular) on hand-built tracking features. Two kinds of model:

- **Team models**: P(home has the ball | ball alive) for every 0.2 s block (5 frames). Eight
  members: {LightGBM, Kumo} × {column set `b`, `ab`} × {label rule R2, rule E}.
- **Dead-ball models**: P(ball alive) for every 12-frame (0.48 s) window. Two models, each a
  50/50 logit blend of a LightGBM part and a Kumo part.

Every member answers a row and its home/away-mirrored copy and averages the two, so neither team
is favoured for being listed first.

```
tabular/
├── src/                     feature and table code: the same code built the training tables and
│   │                        builds BAR-ATH's, so training and inference share it
│   ├── pipeline.py          per-frame tracking features, datasets, windows
│   ├── tabular.py           the per-second tabular row and the symmetric LightGBM fitters
│   ├── context_v3.py        context columns (homography, broadcast) for the tabular row
│   └── lab/exp_*/           one folder per experiment track; the model tables and training scripts
├── inference/               the pipeline's two stages for these models
│   ├── tabular_features.py  feature tables of BAR-ATH, built with src/
│   ├── tabular_models.py    loads the weights and predicts
│   └── kumo.py              runs one Kumo member (GPU, or CPU with chunked attention)
├── training/                training helpers (see "Training")
└── weights/
    ├── team_idsse_rule_R2/            lightgbm_{b,ab}.txt, kumo_{b,ab}.context.npz / .columns.npy, features_{b,ab}.json
    ├── team_idsse_rule_E/             the same, trained with rule-E labels
    ├── dead_ball_idsse_skillcorner/   lightgbm.txt (+ .meta.json), kumo.context.npz / .columns.npy, features.json
    ├── dead_ball_idsse_carried_ball/  the same
    └── kumo_checkpoint/               NVIDIA's network, downloaded by the pipeline (not in git)
```

## Model

- **LightGBM**: team members 800 trees with 63 leaves; dead-ball parts 300 trees with 31 leaves.
  The `.txt` files are the boosters (`lightgbm.Booster(model_file=...)`).
- **Kumo Tabular large** (`nvidia/Kumo-Tabular`): a pretrained tabular foundation model used
  **in context only**; nothing in it is trained. A member is fully described by its context: the
  50,000 labelled training rows (random sample, seed 0) handed to `fit`, on the 100 columns that
  LightGBM ranked highest. The `.context.npz` files hold those rows and the `.columns.npy` files
  the column indices; with the pretrained checkpoint, they are the fitted model.
- **Columns.** Team set `b`: ball velocity and acceleration, a touch detector, the last-touch team,
  players within 3/5/10 m of the ball, team shape, plus the per-second tabular row. Set `ab`: `b`
  plus control and touch columns (control shares at 1/1.5/2.5 m, touches, time since and until
  each team's last and next touch). Dead-ball windows: the tabular row, the same raw-position
  columns at 12-frame windows, and player-context columns. The exact column lists are in the
  `features*.json` files.

## Training data

All labelled data is open; no BAR-ATH labels are used. (BAR-ATH's tracking is used without labels,
see "Training".)

| name | what |
|---|---|
| IDSSE | 7 Bundesliga / 2. Bundesliga matches of the IDSSE open dataset (DFL, CC BY 4.0): J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9 |
| `d85su` | IDSSE made to look like BAR-ATH's automatic tracking: DFL's frozen dead ball unfrozen, the ball hidden mostly during stoppages (85% of the hidden time), the ball smoothed (σ = 6 frames) |
| `d85suc` | `d85su` relabelled to **controlled** possession, the challenge's convention: rule R2 (k = 0.6, matched to SkillCorner's possession statistics) |
| `d85sue` | `d85su` with rule-E labels (DFL's events decide whether a change of possession was controlled; differs from R2 on 3.25% of live frames) |
| `r85suc` | `d85suc` with a realistic ball during stoppages (`carry.py`: rolled out, picked up, carried to the taker and placed) instead of IDSSE's synthetic one |
| SkillCorner `clean` | 20 matches of SkillCorner's open broadcast tracking, converted to 25 fps, ball smoothed (σ = 4); labels from SkillCorner's own phases of play (a team inside a phase, dead outside) |

| weights | trained on |
|---|---|
| `team_idsse_rule_R2` | IDSSE `d85suc` + SkillCorner `clean` |
| `team_idsse_rule_E` | IDSSE `d85sue` + SkillCorner `clean` |
| `dead_ball_idsse_skillcorner` | IDSSE `d85suc` + SkillCorner `clean` |
| `dead_ball_idsse_carried_ball` | IDSSE `d85suc` + `r85suc` |

## Inference

The pipeline runs these models in two stages:

- `tabular_features` (`inference/tabular_features.py`) runs `src/pipeline.py features` and
  `src/lab/exp_t/build_bar.py`, then the table code of `src/lab/exp_k/bar_k.py` (team blocks,
  dead-ball windows, kick-off mask);
- `tabular_models` (`inference/tabular_models.py`, with `inference/kumo.py`) loads the weights and predicts.

The feature and model calculations retain Ragha's original implementation. Packaging changes
repair module/scorer paths and pin the training downloaders to `training_data.lock.json`.
The paired-bootstrap utility is now `src/paired_bootstrap.py`; the official scorer is included
at `../../challenge/evaluation/run_evaluation.py`. These utilities take external validation
labels; they are not used by submission inference. Intermediate files remain in
`models/tabular/runs/` (gitignored).

The same seven IDSSE and twenty SkillCorner source matches also underlie the physics calibration.
Its converted input schema differs; the tabular downloader does not recreate that conversion.
The BiLSTM uses a subset of the IDSSE matches. Source revisions and complete match IDs are fixed
in the shared manifest. SkillCorner cache entries are isolated by revision.

## Training

These are the commands the weights were made with (RunPod, 14 November 2026). The data
preparation steps had been run earlier with the same code. Run them from `models/tabular/`; every
path below is relative to it.

```bash
export MTAH_ROOT=$(pwd)                       # the code finds runs/ and src/ from here
export MTAH_DATA_DIR=/path/to/challenge/data  # the provided BAR-ATH data
export PYTHONPATH=$MTAH_ROOT/../../challenge:$MTAH_ROOT/../../challenge/starter:$MTAH_ROOT/../../challenge/evaluation
export KLOPPY_CACHE_DIR=/path/to/empty/dir    # extract_tb.py EMPTIES this folder after each match
export SC_CACHE=/path/to/sc_cache OMP_NUM_THREADS=8
# First run `uv sync --locked --group training` from the repository root.
source ../../.venv/bin/activate
```

**1. Data preparation** (writes `runs/`; IDSSE and SkillCorner are downloaded)

| step | command | builds |
|---|---|---|
| 1 | `python src/pipeline.py features` | BAR-ATH per-frame features |
| 2 | `python src/lab/exp_t/build_bar.py` | BAR-ATH raw positions |
| 3 | `python src/idsse.py --match $m` for each of the 7 IDSSE matches | `runs/idsse-<m>/` (downloads from the `pysport/idsse-data` mirror on Hugging Face) |
| 4 | `python src/tabular.py degrade --dead-share 0.85` | `idsse-<m>-d85`: ball hidden as in BAR-ATH |
| 5 | `python src/lab/exp_b/extract_tb.py $m` for each match | raw positions of the clean and degraded matches |
| 6 | `python src/lab/exp_u/unfreeze.py` | dead ball unfrozen |
| 7 | `python src/lab/exp_u/smooth_u.py --sigma 6` | `idsse-<m>-d85su` |
| 8 | `python src/lab/exp_sc/sc_build.py` | the 20 SkillCorner matches, `clean` (downloads from SkillCorner's open-data GitHub repository) |
| 9 | `python src/lab/exp_sc/conv_sc.py` | `idsse-<m>-d85suc`: rule-R2 controlled-possession labels (chooses k = 0.6 against SkillCorner) |
| 10 | `python src/lab/exp_bench/labels.py lab --cond d85su` | `idsse-<m>-d85sue`: rule-E labels (downloads DFL's event files from the same mirror) |
| 11 | `python src/lab/exp_dball/carry.py` | `idsse-<m>-r85suc`: the carried stoppage ball |

The BAR-ATH tracking is used without labels: step 4 copies its ball-gap lengths, step 8 chooses
the ball smoothing against it, and the training scripts below also predict on it.

**2. Training.** The lab scripts keep only predictions, so each runs under
`training/wcap.py`, which runs a script unchanged and saves every model it fits: LightGBM
boosters, Kumo contexts and the chosen columns, in fit order (`OUTDIR/TAG_<n>_<kind>`). The Kumo
steps need a CUDA GPU; LightGBM is deterministic for a fixed number of threads (`MTAH_LGBM_JOBS`).

```bash
W="python training/wcap.py"
# team_idsse_rule_R2 (GPU)
MTAH_LGBM_JOBS=8 $W out/team_R2 team_R2 src/lab/exp_k/bar_k3.py members --train both
# team_idsse_rule_E (GPU)
MTAH_IDSSE_COND=d85sue MTAH_LGBM_JOBS=12 $W out/team_E team_E src/lab/exp_k/bar_k3.py members --train both
# dead_ball_idsse_skillcorner (GPU)
MTAH_LGBM_JOBS=8 $W out/palive_both palive_both -c "import sys; sys.path.insert(0, 'src/lab/exp_k'); \
  import bar_k2; bar_k2.TRAIN.update(bar_k2.train_set('both')); bar_k2.palive_mix('exp_k_all_pa_both')"
# dead_ball_idsse_carried_ball: LightGBM part (CPU), then Kumo part (GPU)
MTAH_LGBM_JOBS=8 $W out/palive_R1 palive_R1 src/lab/exp_dball/dpa.py --name R1 \
  --train d85suc r85suc --feats pf --no-lomo --evals d85suc --xfer --bar
MTAH_LGBM_JOBS=8 $W out/palive_R1k palive_R1k src/lab/exp_dball/dpa.py --name R1k --model kumo \
  --train d85suc r85suc --feats pf --no-lomo --evals d85suc --xfer --bar
```

From the captures to `weights/`: in a team run, the two LightGBM boosters fitted by `fit_team`
are `lightgbm_b.txt` then `lightgbm_ab.txt`, and the two Kumo contexts with their column
selections are `kumo_b` then `kumo_ab`. In a dead-ball run, the LightGBM booster that is not a
column ranker is `lightgbm.txt`, and the Kumo context with its columns is `kumo`. The
column-ranking boosters are only used to choose the Kumo columns and are not needed to predict.
`features*.json` hold the column names of the tables the models were fitted on.
