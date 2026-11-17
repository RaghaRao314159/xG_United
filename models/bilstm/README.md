# Two-head BiLSTM (Tiya)

A recurrent network that reads 30 tracking features per frame and gives, for every frame,
P(home), P(away) and P(dead). The ensemble uses it twice: its home/away probabilities (from a run
on home/away-exchanged inputs) as team evidence, and its own decoder's labels as a second opinion
on dead ball.

```
bilstm/
├── inference/
│   ├── run_inference.py         network definition, windowed batching and Tiya's Viterbi decoder
│   └── bilstm_model.py          the pipeline stage: builds the 30 features, runs the network
├── training/                    data preparation, training and decoder selection
└── weights/model_weights.pt     the trained network (state dict, input mean/scale, config)
```

## Model

- **Inputs ("schema 3"), 30 per frame**, on a 105 × 68 m reference pitch: ball x/y, velocity x/y,
  speed and a ball-visible flag; the 3 nearest home and 3 nearest away players' distances to the
  ball; for each team the number of tracked players, centroid x/y, spread x/y, centroid velocity
  x/y and the median and 75th percentile of player speeds. No provider speed, height, tactical
  tags or labels are inputs.
- **Network**: a three-layer bidirectional LSTM, hidden size 256, with two heads: *dead*
  (binary: ball dead or alive) and *team given alive* (binary: home or away). Windows of 512
  frames with 256 frames of context on each side.
- **Decoders** (temporal smoothing of the frame probabilities, used by the ensemble): Viterbi
  with switching costs home↔away / into dead / out of dead of (4, 4, 4) and (1, 1, 6). The second
  was chosen on the development match only.

## Training data

Open IDSSE matches (DFL Bundesliga and 2. Bundesliga, CC BY 4.0).
The public mirror revision and source attribution are in `training_data.lock.json` at the
repository root; the downloader uses that immutable revision. The labels are the provider's own per-frame
ball status and ball-owning team (`D` where the ball is dead, otherwise the owning team):

| role | matches |
|---|---|
| training | J03WPY, J03WMX, J03WOH, J03WQQ, each twice: as recorded and with synthetic player-tracking gaps |
| development (checkpoint and decoder selection) | J03WN1 (and its synthetic-gap copy) |
| reserved, never loaded | J03WR9 |

No BAR-ATH labels were used to train the network or to choose its decoders.

## Inference

The pipeline runs it as stage `bilstm_model` (`inference/bilstm_model.py`), which builds the 30
features from the provided tactical tracking and calls `inference/run_inference.py`, unchanged
from Tiya's hand-off. Stand-alone, on a prepared feature file:

```bash
python models/bilstm/inference/run_inference.py --input inputs.csv.gz --out labels.csv \
  --weights models/bilstm/weights/model_weights.pt
```

## Training

On a CUDA GPU, from `models/bilstm/training/`:

```bash
# First run `uv sync --locked --group training` from the repository root.
# Activate that environment before changing into this folder:
source ../../../.venv/bin/activate
export PYTHONPATH=../../../challenge/evaluation      # the challenge's metrics, for development scores
# 1. download the IDSSE matches and build the training folders (10 folders, roughly half an hour)
python download_idsse.py --cache <cache dir> --output <input-root>
# 2. train
python train_twohead_v3.py --input-root <input-root> --run-root <runs dir> --model lstm --epochs 60 --seed 42
# 3. choose the decoder's transition costs on the development match's predictions
python select_asymmetric_decoder.py --run <run folder from step 2> --evaluation ../../../challenge/evaluation
```

| file | role |
|---|---|
| `download_idsse.py` | downloads the five IDSSE matches with kloppy and builds each one's training folders, as recorded and with synthetic player gaps |
| `prepare_model_inputs.py` | ball features and the IDSSE / Metrica row readers |
| `prepare_model_inputs_v3.py` | schema-3 features (team movement) and the synthetic player gaps |
| `train_twohead_v3.py` | trains the two-head network, logs development scores |
| `possession_decode.py` | the Viterbi decoder used during training and selection |
| `select_asymmetric_decoder.py` | picks the asymmetric decoder's costs on development predictions |

**The training data is reproducible exactly.** At training time, `train_twohead_v3.py` stores in
the checkpoint the SHA-256 of every training folder's files, the input normalisation statistics
and the development scores. The folders `download_idsse.py` builds match all of them: the
`summary.json` of all 10 folders hash-identical, the normalisation statistics over the 1,143,474
training frames bit-identical, and the saved network's development scores on J03WN1 (macro F1,
spell F1 and possession counts, with and without synthetic gaps) identical.
