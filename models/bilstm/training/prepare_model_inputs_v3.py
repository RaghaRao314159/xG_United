"""Tracking schema 3: shared ball inputs plus ball-independent team movement."""
import csv
import gzip
import json
import math
from pathlib import Path
import random

from prepare_model_inputs import (
    FEATURES as BALL_FEATURES, Features as BallFeatures, FPS, LENGTH, WIDTH,
    INDEX, idsse_rows, metrica_rows,
)

TEAM_FIELDS = ('count', 'centroid_x_ref_m', 'centroid_y_ref_m', 'spread_x_ref_m',
               'spread_y_ref_m', 'centroid_vx_ref_mps', 'centroid_vy_ref_mps',
               'median_speed_ref_mps', 'p75_speed_ref_mps')
FEATURES = BALL_FEATURES + [f'{side}_{field}' for side in ('home', 'away') for field in TEAM_FIELDS]


def quantile(values, q):
    if not values:
        return float('nan')
    values = sorted(values)
    position = q * (len(values) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] * (hi - position) + values[hi] * (position - lo) if lo != hi else values[lo]


def shape(points):
    points = [p for p in points if p is not None]
    if not points:
        return [0] + [float('nan')] * 4
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    sx = math.sqrt(sum((p[0] - cx)**2 for p in points) / len(points))
    sy = math.sqrt(sum((p[1] - cy)**2 for p in points) / len(points))
    return [len(points), cx, cy, sx, sy]


class Features(BallFeatures):
    def __init__(self):
        super().__init__()
        self.previous_players = None

    def build(self, frame_id, period, ball, players):
        adjacent = self.previous is not None and self.previous[:2] == (frame_id - 1, period)
        result = super().build(frame_id, period, ball, players)
        for side in ('home', 'away'):
            current = shape(players[side])
            vx = vy = float('nan')
            speeds = []
            if adjacent and self.previous_players is not None:
                previous = self.previous_players[side]
                assert len(previous) == len(players[side])
                previous_shape = shape(previous)
                if previous_shape[0] and current[0]:
                    vx = (current[1] - previous_shape[1]) * FPS
                    vy = (current[2] - previous_shape[2]) * FPS
                speeds = [math.hypot(now[0] - before[0], now[1] - before[1]) * FPS
                          for now, before in zip(players[side], previous)
                          if now is not None and before is not None]
            # Quantiles reduce the effect of occasional slot identity switches.
            result.extend(current + [vx, vy, quantile(speeds, .5), quantile(speeds, .75)])
        self.previous_players = players
        assert len(result) == len(FEATURES) == 30
        assert not any(math.isinf(value) for value in result)
        return result


def player_gaps(rows, seed=123):
    """Synthetic persistent track gaps, applied BEFORE feature extraction."""
    rng = random.Random(seed)
    previous_period = None
    hidden = {}
    for frame_id, period, ball, players, label in rows:
        if previous_period != period:
            hidden = {side: [False] * len(points) for side, points in players.items()}
            previous_period = period
        output = {}
        for side, points in players.items():
            assert len(points) == len(hidden[side])
            for i in range(len(points)):
                if hidden[side][i]:
                    if rng.random() < .02:
                        hidden[side][i] = False
                elif rng.random() < .005:
                    hidden[side][i] = True
            output[side] = [None if drop else p for p, drop in zip(points, hidden[side])]
        yield frame_id, period, ball, output, label


def export(rows, out, source, dropout=False):
    out.mkdir(parents=True, exist_ok=False)
    builder = Features()
    targets = gzip.open(out / 'targets.csv.gz', 'wt', newline='') if source == 'idsse' else None
    count = missing_ball = missing_speed = 0
    visible_players = 0
    try:
        target_writer = csv.writer(targets) if targets else None
        if target_writer:
            target_writer.writerow(['frame', 'label'])
        with gzip.open(out / 'inputs.csv.gz', 'wt', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(INDEX + FEATURES)
            for frame, (frame_id, period, ball, players, label) in enumerate(rows):
                values = builder.build(frame_id, period, ball, players)
                writer.writerow([frame, period, frame_id] + values)
                if target_writer:
                    assert label in ('H', 'A', 'D')
                    target_writer.writerow([frame, label])
                else:
                    assert label is None
                missing_ball += not values[5]
                missing_speed += math.isnan(values[4])
                visible_players += values[12] + values[21]
                count += 1
    finally:
        if targets:
            targets.close()
    assert count
    summary = {'schema': 3, 'source': source, 'fps': FPS,
               'reference_pitch_metres': [LENGTH, WIDTH], 'features': FEATURES,
               'frames': count, 'missing_ball_frames': missing_ball,
               'missing_derived_speed_frames': missing_speed,
               'mean_visible_players': visible_players / count,
               'synthetic_player_gaps': dropout,
               'height_used': False, 'provider_speed_used': False, 'complete': True}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Saved schema 3: {out.name}; {count:,} frames; '
          f'mean visible players {visible_players / count:.2f}', flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(metrica_rows(args.data_dir), args.output, 'metrica')


if __name__ == '__main__':
    main()
