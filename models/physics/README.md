# Physics model v9d (Narmee)

A rule-based model: football physics and kinematics on the tracking, ARIMA / state-space filters,
Smart Tagging and the broadcast camera's tracking, decoded per half with a minimum-duration
Viterbi. It gives a label per frame. The ensemble uses it only as a second opinion on dead ball.

```
physics/
├── physics_rules/                         the model, a Python package (Narmee's code, unchanged)
│   ├── v9.py                              pro_v9d_ultimate_arimax_physics: the model the pipeline runs
│   ├── v8.py, methods.py                  the rules and decoders v9 builds on
│   ├── features.py, data.py               per-frame features; loading a match in metres and frames
│   └── arima.py                           ARIMA / ARIMAX signals, and the cross-match fit (training)
├── inference/physics_model.py             the pipeline stage: runs v9d on the provided data
└── weights/cross_match_arima_learned.json the only numbers v9d learned
```

## Model

`pro_v9d_ultimate_arimax_physics` (`v9.py`) combines three rule sets:

- **v9a**: goalkeeper penalty-box invariance, filtering of frozen phantom tracklets, fusion of the
  tactical and broadcast cameras' player proximity, and pass-flight continuation to a teammate.
- **v9b**: bridges unseen aerial passes (ball lost when in the air, chord speed 2.5–30 m/s,
  players running) by attributing the flight to the kicking team.
- **v9c**: a constant-velocity Kalman filter (ARIMA(0,2,2)) with volatility normalisation, plus a
  last-touch out-of-bounds restart rule.

It then snaps possession boundaries to contacts and decodes with a minimum-duration semi-Markov
Viterbi. The ARIMA filters are fitted on the match's own tracking (no labels involved).

## Learned numbers (training)

`weights/cross_match_arima_learned.json` holds the only learned parameters: a ridge calibration and
spell-rate transition scales. They were fitted with 5-fold out-of-fold cross-validation, grouped by
match, on 27 open labelled matches: 7 IDSSE (DFL Bundesliga) and 20 SkillCorner (A-League). No
BAR-ATH labels were used.

The fit is `arima._fit_or_load_cross_match_learner(refresh=True)`. It reads the 27 matches from
`<data folder>/external/converted/<GAME>/` (`<GAME>_meta.json` plus tracking, see
`data.load_match`) and rewrites the JSON above:

```bash
cd models/physics
PYTHONPATH=../../challenge:../../challenge/starter:../../challenge/evaluation MTAH_DATA_DIR=<data folder> \
  python -c "from physics_rules.arima import get_cross_match_learned_params as fit; fit('BAR-ATH', refresh=True)"
```

The seven IDSSE match IDs and twenty SkillCorner match IDs exactly match the tabular model's
source-match set; see `training_data.lock.json` at the repository root. Ragha's scripts and
Tiya's IDSSE downloader provide access to those public sources at pinned revisions.

**Training reconstruction limit.** The physics learner expects a separate converted EPTS-style
layout (metadata, tracking, possession labels and optional tags), whereas the tabular scripts
write NumPy feature tables. The original conversion script was not included in Narmee's hand-off.
Sharing source matches does not make those prepared inputs interchangeable. The bundled
`weights/cross_match_arima_learned.json` is sufficient for submission inference; reproducing
that calibration from raw external data still requires the original conversion or a separately
verified equivalent. No claim of identical retraining is made.

## Inference

The pipeline runs it as stage `physics_model` (`inference/physics_model.py`). It needs the
tactical tracking, its metadata and Smart Tagging, plus the broadcast tracking. The code skips the
broadcast camera silently when those files are missing, which would change the labels, so the
stage refuses to run without them.
