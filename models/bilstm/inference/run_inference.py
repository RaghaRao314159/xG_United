"""Run the Tiya two-head possession model on a prepared schema-3 input CSV.

Example:
  python run_inference.py --input metrica-v3/inputs.csv.gz --out submission.csv

The input must contain frame, period_id, frame_id and the 30 schema-3 feature
columns. The checkpoint supplies the normalisation statistics. The default
decoder is the externally selected asymmetric H/A/D decoder.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch


FEATURES = [
    'ball_x_ref_m', 'ball_y_ref_m', 'ball_vx_ref_mps', 'ball_vy_ref_mps',
    'ball_speed_ref_mps', 'ball_visible',
    'home_distance_1_ref_m', 'home_distance_2_ref_m', 'home_distance_3_ref_m',
    'away_distance_1_ref_m', 'away_distance_2_ref_m', 'away_distance_3_ref_m',
    *[f'{side}_{field}' for side in ('home', 'away') for field in (
        'count', 'centroid_x_ref_m', 'centroid_y_ref_m', 'spread_x_ref_m',
        'spread_y_ref_m', 'centroid_vx_ref_mps', 'centroid_vy_ref_mps',
        'median_speed_ref_mps', 'p75_speed_ref_mps')],
]
LABELS = np.array(['H', 'A', 'D'])
CORE, HALO = 512, 256


class TwoHeadLSTM(torch.nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(len(FEATURES) * 2, 128),
                                           torch.nn.GELU())
        self.memory = torch.nn.LSTM(128, hidden, num_layers=3, batch_first=True,
                                    bidirectional=True, dropout=.25)
        self.dead_head = torch.nn.Sequential(torch.nn.Linear(hidden * 2, 128),
                                             torch.nn.GELU(), torch.nn.Linear(128, 1))
        self.team_head = torch.nn.Sequential(torch.nn.Linear(hidden * 2, 128),
                                             torch.nn.GELU(), torch.nn.Linear(128, 2))

    def forward(self, x):
        sequence, _ = self.memory(self.encoder(x))
        return self.dead_head(sequence), self.team_head(sequence)


def contiguous_spans(frame):
    starts = np.flatnonzero((frame.period_id.ne(frame.period_id.shift()) |
                             frame.frame_id.diff().ne(1)).to_numpy())
    return list(zip(starts.tolist(), np.r_[starts[1:], len(frame)].tolist()))


def windows(spans):
    return [(lo, hi, start) for lo, hi in spans for start in range(lo, hi, CORE)]


def normalize(values, mean, scale):
    missing = np.isnan(values)
    values = np.clip(np.nan_to_num((values - mean) / scale), -20, 20)
    return np.concatenate((values, missing.astype(np.float32)), axis=-1).astype(np.float32)


def asymmetric_decode(probabilities, spans, team_switch, to_dead, from_dead):
    """Viterbi decoder with separately selected state-transition costs."""
    penalty = np.array([[0., team_switch, to_dead], [team_switch, 0., to_dead],
                        [from_dead, from_dead, 0.]])
    emissions = np.log(np.maximum(probabilities, 1e-12))
    output = np.empty(len(probabilities), dtype=np.int8)
    for lo, hi in spans:
        scores = emissions[lo].copy()
        back = np.empty((hi - lo - 1, 3), dtype=np.int8)
        for step, row in enumerate(emissions[lo + 1:hi]):
            candidates = scores[:, None] - penalty
            previous = candidates.argmax(axis=0)
            scores = candidates[previous, np.arange(3)] + row
            scores -= scores.max()
            back[step] = previous
        state = int(scores.argmax())
        output[hi - 1] = state
        for frame_index in range(hi - 2, lo - 1, -1):
            state = int(back[frame_index - lo, state])
            output[frame_index] = state
    return output


def predict(model, values, spans, mean, scale, device, batch_size):
    probabilities = np.full((len(values), 3), np.nan, dtype=np.float32)
    all_windows = windows(spans)
    model.eval()
    with torch.inference_mode():
        for first in range(0, len(all_windows), batch_size):
            chosen = all_windows[first:first + batch_size]
            index = np.stack([np.clip(start + np.arange(-HALO, CORE + HALO), lo, hi - 1)
                              for lo, hi, start in chosen])
            x = torch.from_numpy(normalize(values[index], mean, scale)).to(device)
            dead_logits, team_logits = model(x)
            dead = torch.sigmoid(dead_logits[:, HALO:HALO + CORE])
            team = torch.softmax(team_logits[:, HALO:HALO + CORE], dim=-1)
            result = torch.cat(((1. - dead) * team, dead), dim=-1).cpu().numpy()
            for row, (_, hi, start) in enumerate(chosen):
                count = min(CORE, hi - start)
                probabilities[start:start + count] = result[row, :count]
    assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    return probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--weights', type=Path, default=here.parent / 'weights' / 'model_weights.pt')
    parser.add_argument('--input', type=Path, required=True, help='schema-3 inputs.csv.gz')
    parser.add_argument('--out', type=Path, default=Path('submission.csv'))
    parser.add_argument('--probabilities', type=Path, help='optional CSV.GZ probability output')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--team-switch', type=float, default=1.)
    parser.add_argument('--to-dead', type=float, default=1.)
    parser.add_argument('--from-dead', type=float, default=6.)
    args = parser.parse_args()
    if min(args.team_switch, args.to_dead, args.from_dead) < 0:
        raise ValueError('decoder costs must be non-negative')
    frame = pd.read_csv(args.input)
    expected = ['frame', 'period_id', 'frame_id'] + FEATURES
    if frame.columns.tolist() != expected:
        raise ValueError('input columns do not match schema 3')
    if not np.array_equal(frame.frame.to_numpy(), np.arange(len(frame))):
        raise ValueError('frame must be contiguous and zero-based')
    values = frame[FEATURES].to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError('input contains infinity')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    if checkpoint['config'].get('heads') != {'dead': 'binary', 'team_given_alive': 'binary'}:
        raise ValueError('weights are not the expected two-head checkpoint')
    model = TwoHeadLSTM(hidden=checkpoint['config']['hidden']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    spans = contiguous_spans(frame)
    probabilities = predict(model, values, spans, np.asarray(checkpoint['mean'], dtype=np.float32),
                            np.asarray(checkpoint['scale'], dtype=np.float32), device, args.batch_size)
    labels = asymmetric_decode(probabilities, spans, args.team_switch, args.to_dead, args.from_dead)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['frame', 'label'])
        writer.writerows(enumerate(LABELS[labels]))
    if args.probabilities:
        table = pd.DataFrame({'frame': frame.frame, 'prob_H': probabilities[:, 0],
                              'prob_A': probabilities[:, 1], 'prob_D': probabilities[:, 2]})
        table.to_csv(args.probabilities, index=False, compression='gzip')
    print(f'Wrote {args.out} ({len(frame):,} frames) on {device}.')


if __name__ == '__main__':
    main()
