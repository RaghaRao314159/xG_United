# xG United

## Frame-level ball possession

**More than a Hack 2027 - Phase 1 technical document**

### 1. Approach and competition assets

Our system assigns one label to every frame of the BAR-ATH tactical video: home possession (`H`), away possession (`A`), dead ball (`D`), or outside the match's graded halves (`X`). The output is `submission/xG United.csv`, with 199,319 rows indexed from 0 to 199318.

We combine three complementary models. Tabular models summarize local tracking evidence and learn from external matches. A bidirectional recurrent network supplies longer temporal context. A physics model supplies an independent alive/dead opinion. A temporal decoder converts their probabilities into possession intervals. This combination addresses the central difficulty of automatic tracking: the nearest visible player is not always the player controlling the ball, and missing ball observations can occur during either live play or stoppages.

| Component | Evidence supplied | Role in the final ensemble |
|---|---|---|
| Ragha's LightGBM and Kumo Tabular models | Team given alive, and ball alive probabilities | Primary local team and dead-ball evidence |
| Tiya's two-head BiLSTM | Home, away and dead probabilities from tracking sequences | Temporal team evidence and an alive/dead opinion |
| Narmee's physics model v9d | Labels from kinematics, filters and automatic tags | Additional live-ball evidence when the BiLSTM agrees |

The final pipeline uses the following competition assets:

| Asset | Use |
|---|---|
| Tactical ATD text and XML | Ball/player positions, team assignment, pitch and frame alignment |
| Tactical homography CSV | On-screen player masks for the tabular feature pipeline |
| Tactical Smart Tagging XML | Automatic contextual evidence for the physics model |
| Broadcast ATD text and XML | Second-camera tracking evidence in the physics model |
| Published match constants | Half boundaries and kick-off anchors supplied by the organisers |

Raw video pixels and broadcast homography are not inputs to the final pipeline. The final system does not run a detector, a video-language model or a hosted inference API. The public Kumo checkpoint is downloaded once; subsequent model inference runs locally. Earlier experimental helpers retained in the source tree are not additional submission stages.

The six execution stages are weight verification, tabular feature construction, tabular prediction, BiLSTM prediction, physics prediction and final ensemble decoding. Each runs in its own process. Intermediate outputs are regenerated in ignored work directories.

---

### 2. Training data and provenance

The documented supervised training sources are external matches from **IDSSE** and **SkillCorner Open Data**. The tabular and physics artifacts list the same seven IDSSE and twenty SkillCorner match identities. Their prepared arrays and label transformations differ; sharing source matches does not make their training inputs interchangeable. The BiLSTM uses a subset of IDSSE.

| Source | Amount and origin of labels | Models using it |
|---|---|---|
| IDSSE, DFL / Sportec Solutions | Seven Bundesliga and 2. Bundesliga matches. Provider `BallStatus` and `BallPossession` supply dead/alive and team targets; provider events also support tabular rule-E labels. | Tabular and physics; a subset for BiLSTM |
| SkillCorner Open Data | Twenty A-League matches. The tabular conversion uses the provider's phases of play: the named team inside a phase, dead outside phases within each playing period. | Tabular and physics |
| Synthetic variants | Algorithmic tracking gaps, smoothing, ball-motion variants and controlled-possession relabelling derived from the external matches. | Tabular and BiLSTM |

**IDSSE match IDs:** J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ and J03WR9. IDSSE is distributed under CC BY 4.0. Attribution: Bassek, Rein, Weber and Memmert, *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12, 195 (2025), [paper and original dataset reference](https://doi.org/10.1038/s41597-025-04505-y). The scripts download from the [PySport IDSSE mirror](https://huggingface.co/datasets/pysport/idsse-data).

**SkillCorner match IDs:** 1874553, 1886347, 1899585, 1925299, 1927964, 1953632, 1959846, 1986691, 1996435, 1996436, 2006229, 2006363, 2007448, 2007721, 2010085, 2011166, 2013725, 2015213, 2016236 and 2017461. Credit: [SkillCorner Open Data](https://github.com/SkillCorner/opendata), released with PySport. The external labels are provider-supplied; the transformations described below are implemented in code.

The shared `training_data.lock.json` fixes these IDs and the download revisions: IDSSE `a715a38dfbaf5f58e431727c2b78d174101a703c`, and SkillCorner `4340d274572876239c154c90bc507a9b3250a656`. The tabular and BiLSTM downloaders use those revisions. The lock was added during packaging and identifies the sources available for reconstruction.

**Tabular preparation.** The seven IDSSE matches are degraded to resemble automatic tracking: ball gaps sampled from unlabelled BAR-ATH tracking, dead-ball unfreezing, and smoothing with sigma 6 frames. Rule R2 requires opposing control within 1.5 m for 0.6 seconds; its threshold was selected against SkillCorner possession statistics. Rule E instead uses DFL events to confirm controlled changes. SkillCorner tracking is resampled from 10 to 25 fps and smoothed with sigma 4. A second dead-ball training variant simulates balls being retrieved, carried and placed before restarts. These are algorithmic transformations of external data.

**BiLSTM split.** Training uses J03WPY, J03WMX, J03WOH and J03WQQ; development uses J03WN1. Each has a clean copy and a seeded player-gap copy. The training set contains 1,143,474 frames across those copies. J03WR9 is reserved and not loaded by this model's training. The checkpoint contains normalization and training-data hash records.

---

### 3. Models and possession rules

**Tabular models.** Eight team members cross two model families (LightGBM and Kumo), two feature sets (`b`, `ab`) and two external label rules (R2, E). Each predicts home possession given a live ball for a five-frame block. Features include ball motion, player proximity, touches, team shape and control history. Two separate dead-ball models predict alive probability over twelve-frame windows; each blends LightGBM and Kumo logits equally. One uses IDSSE plus SkillCorner, the other IDSSE plus carried-ball variants.

Kumo Tabular large is a pretrained in-context model, not fine-tuned here. Each member uses a saved context of 50,000 labelled external-training rows and 100 selected columns, with seed 0 and four estimators. Home/away mirrored predictions reduce sensitivity to which team is listed first. The public network is `nvidia/Kumo-Tabular`, revision `v1.0.1`; its files and all bundled fitted artifacts are SHA-256 checked.

**BiLSTM.** A three-layer bidirectional LSTM with hidden size 256 reads thirty tracking features per frame on a 105 x 68 m reference pitch. It has separate heads for dead/alive and team given alive. Inference uses 512-frame output windows with 256 frames of context on either side. The pipeline runs ordinary and team-exchanged inputs. Its two decoder settings are `(4, 4, 4)` and `(1, 1, 6)` for team switches, entering dead ball and leaving dead ball; the asymmetric setting was selected on the external development match.

**Physics v9d.** Rules combine proximity, goalkeeper context, pass-flight continuation, unseen-flight bridging, out-of-bounds restarts and tactical/broadcast tracking. ARIMA and Kalman-style filters operate on unlabelled match tracking. The supplied cross-match calibration records twenty-seven external matches and five match-grouped folds. Only its alive/dead opinion enters the final ensemble.

**Final possession decisions.** The ensemble averages the eight tabular team logits, adds 1.5 times the BiLSTM team log-odds clipped to +/-8, and adds a 0.5 vote from each BiLSTM decoder. Dead-ball evidence averages the two tabular alive logits; agreement that play is live from both physics and BiLSTM adds up to one alive logit per block. A three-state Viterbi runs separately on each half.

- **Loose balls, deflections and challenges:** a nearest-player change alone does not force a turnover. Controlled-possession training labels, temporal context, switching costs and minimum durations discourage such fragments. This approximates continued possession until an opponent gains control; it cannot guarantee correct interpretation of every duel.
- **Temporal constraints:** team-switch, entering-dead and leaving-dead costs are 1, 2 and 20. A live possession has a minimum of ten five-frame blocks, or two seconds; dead ball has a one-block minimum.
- **Restarts:** published pre-kick-off frames and algorithmically detected kick-off setups are forced dead. A possession following dead ball is delayed by up to 38 frames while retaining at least fifty frames. Its end before a stoppage is extended by up to five frames, leaving at least one dead frame. These are global rules, not individual-frame edits.
- **Outside the halves:** output is `X`. No abstentions occur inside the two official graded halves in the submitted file.

---

### 4. Reproduction and exact output

On 18 November 2026 (Singapore time), a fresh RunPod instance regenerated the submitted CSV **byte for byte in 369 seconds**. Only source, bundled weights and the six input assets were uploaded; no reference CSV or prediction caches were supplied. Both the official format validator and exact SHA-256 check passed.

| Setting | Verified environment |
|---|---|
| GPU / driver | One NVIDIA A40, 46,068 MiB / 595.112.07 |
| OS | Ubuntu 24.04.3 LTS, Linux x86_64 |
| Python / uv | 3.11.13 / 0.13.6 |
| PyTorch / CUDA runtime | 2.15.0+cu132 / 13.2 |
| cuDNN package | 9.26.0.51 |
| CPU threads | 8, on a 9-vCPU allocation |
| Neural precision | CUDA float16 autocast |
| Inference batching | Kumo: 8,192 query rows; BiLSTM: 64 windows |

Base image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; uv creates the separate project environment above. The original recorded A40 run used driver 595.91.07; its OS and Python patch were unrecorded. Full new-run evidence is in `verification/` and `reproduction.json`.

From the repository root, on that GPU setup with the authorised competition data available locally:

```bash
uv sync --locked
uv run --locked python -m main_pipeline \
  --data /path/to/data --device cuda --threads 8 --verify-reference
uv run --locked python challenge/evaluation/validate_submission.py \
  "submission/xG United.csv"
```

Use uv 0.13.6 or newer. The data directory must contain tactical ATD text/XML, tactical homography and Smart Tagging, and broadcast ATD text/XML in the organiser's folder layout. The first run downloads the public Kumo checkpoint, approximately 855 MB. The other fitted artifacts are included. External training datasets are not needed to reproduce inference from those artifacts.

The exact reference is recorded in `reproduction.json`:

```text
File:   submission/xG United.csv
Rows:   199319
MD5:    6b266c5aadc2885a8b3051de0db15386
SHA256: ac2aba453442471bde7e2e34f74e36b58062701b5be6523b22f9b432f4edf6f6
```

`--verify-reference` fails if the generated bytes differ. The same check can be run independently with `python -m main_pipeline.verify_submission`. The uploaded CSV passes the official format validator and contains H: 51,770, A: 30,708, D: 76,379 and X: 40,462 frames. These are output counts, not accuracy measurements.

`pyproject.toml` preserves all recorded direct inference package pins; `uv.lock` also fixes transitive dependencies. Pip-compatible exports are included. `uv sync --locked --group training` installs the training dependencies. The previously unrecorded scikit-learn and lxml versions are now fixed to 1.9.1 and 6.1.3 as reconstruction pins; this does not establish the versions used historically or promise identical retraining.

---

### 5. Limitations and further work

**Tracking failures and domain shift.** Long ball gaps, identity switches, off-screen receivers and crowded challenges can make live play look like dead ball or assign control to the wrong team. IDSSE optical tracking and SkillCorner broadcast tracking differ from the challenge's automatic tactical tracking. Synthetic degradation narrows that difference without fully reproducing it.

**Label semantics.** Provider ball ownership and phases of play do not exactly match the challenge's controlled-possession convention. Rule R2 and rule E address that mismatch differently; combining them reduces dependence on either conversion but does not eliminate systematic errors.

**Boundary bias.** The two-second minimum may remove genuine short possessions. Fixed restart and stoppage delays can improve consistency while shifting individual boundaries incorrectly. A future model could learn restart-specific timing and uncertainty from additional external matches instead of applying one global correction.

**Offline operation.** The bidirectional network uses future context, and Viterbi decodes each half as a whole. This system is intended for offline match analysis. A live product would need causal inference and an explicit latency budget.

**Hardware sensitivity.** CPU execution uses float32 and other GPUs can choose different numerical kernels. Small probability changes near a decoder boundary can change labels. Exact equality is checked against the submitted SHA-256; it is not guaranteed across arbitrary hardware. The lock is a packaging-time dependency snapshot, not a recovered image of the historical machine.

**Physics training reconstruction.** The fitted calibration is supplied and sufficient for submission inference. Its original conversion from public datasets into EPTS-style training inputs was not included in the hand-off. The tabular and BiLSTM scripts download the same source datasets, but do not reproduce that distinct conversion. Identical physics retraining therefore remains unverified. Recovering and validating that conversion is a specific reproducibility improvement.

Further development would prioritize more external matches, possession-aware validation across providers, calibrated uncertainty in ball occlusions, learned restart timing and reproducible numerical execution on the reference hardware. No competition-match data, tracking exports or video clips are included in the public repository. Model source and documentation use the repository's MIT licence; third-party code and data retain their own terms and attribution.

### Submission files and references

The registered team name is **xG United**. The deliverables are `xG United.csv`, this document as `xG United.md` or `xG United.pdf`, and the public repository URL. Spaces are retained in the filenames; shell examples quote them. Repository folder names do not determine the registered team name.

- [Official submission contract](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md)
- [Competition rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md)
- [IDSSE paper and data reference](https://doi.org/10.1038/s41597-025-04505-y)
- [SkillCorner Open Data](https://github.com/SkillCorner/opendata)
- [NVIDIA Kumo Tabular model](https://huggingface.co/nvidia/Kumo-Tabular)
