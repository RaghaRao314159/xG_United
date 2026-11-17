"""Common tracking-only inputs for Metrica and the cached IDSSE matches.

Positions use a standardized 105 x 68 metre pitch for BOTH providers. These are
reference-pitch distances, not a claim about the exact physical pitch dimensions.
Height, supplied speed, tactical tags and answer labels are not input features.
"""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

FPS = 25
LENGTH, WIDTH = 105., 68.
FEATURES = ['ball_x_ref_m', 'ball_y_ref_m', 'ball_vx_ref_mps',
            'ball_vy_ref_mps', 'ball_speed_ref_mps', 'ball_visible'] + [
    f'{side}_distance_{i}_ref_m' for side in ('home', 'away') for i in (1, 2, 3)]
INDEX = ['frame', 'period_id', 'frame_id']


def point(x, y):
    if x is None or y is None or str(x).strip() == '' or str(y).strip() == '':
        return None
    x, y = float(x), float(y)
    return (x * LENGTH, y * WIDTH) if math.isfinite(x) and math.isfinite(y) else None


class Features:
    def __init__(self):
        self.previous = None

    def build(self, frame_id, period, ball, players):
        nan = float('nan')
        vx = vy = speed = nan
        if self.previous is not None:
            previous_id, previous_period, previous_ball = self.previous
            if (ball is not None and previous_ball is not None and
                    period == previous_period and frame_id == previous_id + 1):
                vx = (ball[0] - previous_ball[0]) * FPS
                vy = (ball[1] - previous_ball[1]) * FPS
                speed = math.hypot(vx, vy)
        self.previous = (frame_id, period, ball)
        result = [*(ball if ball is not None else (nan, nan)), vx, vy, speed,
                  int(ball is not None)]
        for side in ('home', 'away'):
            distances = sorted(math.hypot(p[0] - ball[0], p[1] - ball[1])
                               for p in players[side] if p is not None) if ball is not None else []
            result.extend((distances + [nan] * 3)[:3])
        assert len(result) == len(FEATURES)
        assert not any(math.isinf(value) for value in result)
        return result


def metrica_rows(data_dir):
    tactical = data_dir / 'tactical'
    meta = ET.parse(tactical / 'BAR-ATH_tactical_atd.xml').getroot()
    # The supplied starter baseline uses the same EPTS team and slot ordering.
    teams = [team.get('id') for team in meta.iter('Team')]
    assert len(teams) == 2 and len(set(teams)) == 2
    mapping = dict(zip(teams, ('home', 'away')))
    slots = [mapping[player.get('teamId')] for player in meta.iter('Player')]
    assert slots
    with (tactical / 'BAR-ATH_tactical_atd.txt').open() as handle:
        count = 0
        for expected, line in enumerate(handle, 1):
            parts = line.strip().split(':')
            assert len(parts) == 3, 'Unexpected Metrica record format'
            frame_id = int(parts[0])
            assert frame_id == expected, 'Tracking frames are not contiguous from 1'
            groups = parts[1].split(';')
            assert len(groups) == len(slots), 'Track count differs from metadata'
            players = {'home': [], 'away': []}
            for side, group in zip(slots, groups):
                fields = group.split(',')
                assert len(fields) >= 2
                players[side].append(point(fields[0], fields[1]))
            fields = parts[2].split(',')
            assert len(fields) >= 2
            # No manual half boundaries or ground-truth-derived timing constants.
            yield frame_id, 'recording', point(fields[0], fields[1]), players, None
            count += 1
        assert count == 199319, 'Unexpected tactical frame count'


def idsse_rows(cache, match):
    teams = json.loads((cache / f'{match}.json').read_text())
    sides = {team['ground']: team for team in teams if team['ground'] in ('home', 'away')}
    assert set(sides) == {'home', 'away'}
    labels = {team['id']: side[0].upper() for side, team in sides.items()}
    with gzip.open(cache / f'{match}.csv.gz', 'rt', newline='') as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames)
        slots = {side: [p for p in team['players']
                       if f'{p}_x' in columns and f'{p}_y' in columns]
                 for side, team in sides.items()}
        assert all(len(players) >= 3 for players in slots.values())
        previous = None
        for row in reader:
            period, frame_id = row['period_id'], int(row['frame_id'])
            assert previous is None or previous[0] != period or frame_id > previous[1]
            previous = period, frame_id
            players = {side: [point(row[f'{p}_x'], row[f'{p}_y']) for p in ids]
                       for side, ids in slots.items()}
            assert row['ball_state'] in ('alive', 'dead')
            label = 'D' if row['ball_state'] == 'dead' else labels.get(row['ball_owning_team_id'])
            assert label in ('H', 'A', 'D'), 'Unknown external target'
            yield frame_id, period, point(row['ball_x'], row['ball_y']), players, label


def export(rows, out, source):
    out.mkdir(parents=True, exist_ok=False)
    builder = Features()
    missing_ball = missing_speed = count = 0
    targets = gzip.open(out / 'targets.csv.gz', 'wt', newline='') if source == 'idsse' else None
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
                    target_writer.writerow([frame, label])
                else:
                    assert label is None, 'Competition targets must not be read'
                missing_ball += not values[5]
                missing_speed += math.isnan(values[4])
                count += 1
    finally:
        if targets:
            targets.close()
    assert count > 0
    summary = {'schema': 1, 'source': source, 'fps': FPS,
               'reference_pitch_metres': [LENGTH, WIDTH], 'features': FEATURES,
               'frames': count, 'missing_ball_frames': missing_ball,
               'missing_derived_speed_frames': missing_speed,
               'height_used': False, 'provider_speed_used': False,
               'complete': True}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('Saved:', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, choices=('metrica', 'idsse'))
    parser.add_argument('--data-dir', type=Path, help='Metrica bundle root containing tactical/')
    parser.add_argument('--cache-dir', type=Path, help='IDSSE CSV/JSON cache')
    parser.add_argument('--match', help='IDSSE match ID')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    if args.source == 'metrica':
        if not args.data_dir or args.cache_dir or args.match:
            parser.error('metrica needs --data-dir, without --cache-dir or --match')
        rows = metrica_rows(args.data_dir)
    else:
        if not args.cache_dir or not args.match or args.data_dir:
            parser.error('idsse needs --cache-dir and --match, without --data-dir')
        rows = idsse_rows(args.cache_dir, args.match)
    export(rows, args.output, args.source)


if __name__ == '__main__':
    main()
