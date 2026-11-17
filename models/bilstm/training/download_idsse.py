"""Download the open IDSSE matches and build the BiLSTM's training folders.

    python download_idsse.py --cache <cache dir> --output <input root>

1. Each match is read with kloppy from the IDSSE files (DFL, CC BY 4.0) on the public
   `pysport/idsse-data` mirror, keeping the dead-ball frames, and saved in `<cache dir>` as
   `<match>.csv.gz` (kloppy's table of the match: one row per frame, positions on kloppy's 0-1
   pitch, the ball status and the ball-owning team) and `<match>.json` (the two teams: id,
   home or away, player ids). This is the cache `prepare_model_inputs.idsse_rows` reads.
2. Each match becomes two folders under `<input root>`, as `train_twohead_v3.py` reads them:
   `<match>/` (as recorded) and `<match>-player-gaps/` (with synthetic persistent player-track
   gaps, seeded per match), each with `inputs.csv.gz`, `targets.csv.gz` and `summary.json`.

The matches are the four training matches and the development match; the reserved match J03WR9
is never loaded by training. kloppy keeps its own download cache in ~/kloppy_cache (or
KLOPPY_CACHE_DIR).
"""
import argparse
import json
import os
from pathlib import Path

from prepare_model_inputs import idsse_rows
from prepare_model_inputs_v3 import export, player_gaps

# Resolve the submission root even when this script runs from its model folder.
from pathlib import Path as _SourcePath
import sys as _source_sys
_source_sys.path.insert(0, str(_SourcePath(__file__).resolve().parents[3]))
from models.training_sources import IDSSE_MIRROR
MIRROR = IDSSE_MIRROR
COMPETITION = {   # IDSSE match id -> DFL competition id (000001 Bundesliga, 000002 2. Bundesliga)
    'J03WMX': 'DFL-COM-000001', 'J03WN1': 'DFL-COM-000001', 'J03WOH': 'DFL-COM-000002',
    'J03WOY': 'DFL-COM-000002', 'J03WPY': 'DFL-COM-000002', 'J03WQQ': 'DFL-COM-000002',
    'J03WR9': 'DFL-COM-000002',
}
# The four training matches, then the development match. Each match's synthetic player gaps are
# drawn with seed 123 + its position in this list.
MATCHES = ('J03WPY', 'J03WMX', 'J03WOH', 'J03WQQ', 'J03WN1')
GAP_SEED = {match: 123 + position for position, match in enumerate(MATCHES)}


def url(match, kind):
    name = {'meta': 'DFL_02_01_matchinformation_{}_DFL-MAT-{}.xml',
            'tracking': 'DFL_04_03_positions_raw_observed_{}_DFL-MAT-{}.xml'}[kind]
    return MIRROR + name.format(COMPETITION[match], match)


def download(match, cache):
    """Read one match with kloppy and save its table and teams in the cache."""
    from kloppy import sportec
    from kloppy.io import open_as_file

    streams = []
    for kind in ('meta', 'tracking'):
        stream = open_as_file(url(match, kind))
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            raise IOError(f'{url(match, kind)} came back empty; delete kloppy\'s cached copy and retry')
        stream.seek(0)
        streams.append(stream)
    dataset = sportec.load_tracking(meta_data=streams[0], raw_data=streams[1], only_alive=False,
                                    coordinates='kloppy')
    teams = [{'id': team.team_id, 'name': team.name, 'ground': team.ground.value,
              'players': [player.player_id for player in team.players]}
             for team in dataset.metadata.teams]
    dataset.to_df().to_csv(cache / f'{match}.csv.gz', index=False)
    (cache / f'{match}.json').write_text(json.dumps(teams, indent=2))
    print(f'{match}: {len(dataset.records):,} frames cached', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--cache', type=Path, required=True, help='IDSSE CSV/JSON cache')
    parser.add_argument('--output', type=Path, required=True, help='input root for train_twohead_v3.py')
    parser.add_argument('--matches', nargs='+', default=list(MATCHES), choices=MATCHES)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for match in args.matches:
        if not (args.cache / f'{match}.csv.gz').exists():
            download(match, args.cache)
        export(idsse_rows(args.cache, match), args.output / match, 'idsse')
        export(player_gaps(idsse_rows(args.cache, match), seed=GAP_SEED[match]),
               args.output / f'{match}-player-gaps', 'idsse', dropout=True)


if __name__ == '__main__':
    main()
