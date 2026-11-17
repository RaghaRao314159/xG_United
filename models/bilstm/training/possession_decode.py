"""Viterbi decoding of per-frame H/A/D probabilities with a cost for each change of state."""
import math


def decode_segment(emissions, penalty):
    """Maximum log probability path with a cost for each state change."""
    assert penalty >= 0 and len(emissions) > 0
    scores = list(emissions[0])
    back = []
    for row in emissions[1:]:
        previous = [max(range(3), key=lambda k: scores[k] - penalty * (k != state))
                    for state in range(3)]
        next_scores = [scores[k] - penalty * (k != state) + row[state]
                       for state, k in enumerate(previous)]
        # Recentring leaves the best path unchanged and bounds accumulated scores.
        offset = max(next_scores)
        scores = [value - offset for value in next_scores]
        back.append(previous)
    state = max(range(3), key=lambda k: scores[k])
    path = [state]
    for previous in reversed(back):
        state = previous[state]
        path.append(state)
    return path[::-1]


def self_check():
    rows = [[.99, .005, .005]] * 4 + [[.1, .8, .1]] + [[.99, .005, .005]] * 4
    emissions = [[math.log(p) for p in row] for row in rows]
    assert decode_segment(emissions, 0) == [0] * 4 + [1] + [0] * 4
    assert decode_segment(emissions, 2) == [0] * 9
    rows = [[.99, .005, .005]] * 10 + [[.005, .99, .005]] * 10
    assert decode_segment([[math.log(p) for p in r] for r in rows], 2) == [0] * 10 + [1] * 10
    assert decode_segment([[0., -1., -1.]], 2) == [0]



if __name__ == "__main__":
    self_check()
    print("Decoder self-checks passed", flush=True)
