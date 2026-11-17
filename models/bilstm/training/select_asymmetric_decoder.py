"""Choose H/A/D transition costs on external development predictions only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

LABELS = ('H', 'A', 'D')


def spans(frame):
    starts = np.flatnonzero((frame.period_id.ne(frame.period_id.shift()) |
                             frame.frame_id.diff().ne(1)).to_numpy())
    return list(zip(starts.tolist(), np.r_[starts[1:], len(frame)].tolist()))


def decode(probabilities, spans_, costs):
    """Viterbi with separate H↔A, alive→D, and D→alive costs."""
    team, to_dead, from_dead = costs
    penalty = np.array([[0., team, to_dead], [team, 0., to_dead],
                        [from_dead, from_dead, 0.]])
    emissions = np.log(np.maximum(probabilities, 1e-12))
    result = np.empty(len(probabilities), dtype=np.int8)
    for lo, hi in spans_:
        score = emissions[lo].copy()
        back = np.empty((hi - lo - 1, 3), dtype=np.int8)
        for offset, row in enumerate(emissions[lo + 1:hi]):
            candidates = score[:, None] - penalty
            previous = candidates.argmax(axis=0)
            score = candidates[previous, np.arange(3)] + row
            score -= score.max()
            back[offset] = previous
        state = int(score.argmax())
        result[hi - 1] = state
        for index in range(hi - 2, lo - 1, -1):
            state = int(back[index - lo, state])
            result[index] = state
    return result


def score(pm, truth, prediction, spans_):
    boundaries = [lo for lo, _ in spans_[1:]]
    report = pm.evaluate(np.insert(truth, boundaries, pm.X),
                         np.insert(prediction, boundaries, pm.X))
    return {key: report[key] for key in ('spell_f1_mean', 'macro_f1', 'accuracy',
                                         'n_gt_spells', 'n_pred_spells')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--evaluation', required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.evaluation))
    import possession_metrics as pm
    conditions = {}
    for name in ('clean', 'synthetic_gaps'):
        table = pd.read_csv(args.run / f'dev_{name}_predictions.csv.gz')
        truth = table.truth.map({'H': 0, 'A': 1, 'D': 2}).to_numpy(dtype=np.int8)
        probabilities = table[[f'prob_{label}' for label in LABELS]].to_numpy()
        conditions[name] = (truth, probabilities, spans(table))
    grid = (1, 2, 4, 6, 8)
    results = []
    for team in grid:
        for to_dead in grid:
            for from_dead in grid:
                costs = (team, to_dead, from_dead)
                reports = {name: score(pm, truth, decode(probabilities, spans_, costs), spans_)
                           for name, (truth, probabilities, spans_) in conditions.items()}
                results.append({'costs': {'team_switch': team, 'to_dead': to_dead,
                                          'from_dead': from_dead},
                                'selection_score': float(np.mean([r['spell_f1_mean']
                                                                  for r in reports.values()])),
                                'reports': reports})
    results.sort(key=lambda row: row['selection_score'], reverse=True)
    output = {'best': results[0], 'top_10': results[:10], 'conditions': list(conditions)}
    path = args.run / 'asymmetric_decoder_selection.json'
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output['best'], indent=2))
    print(path)


if __name__ == '__main__':
    main()
