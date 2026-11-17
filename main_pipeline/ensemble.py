"""Stage 6: combine the three models into the submission file.

Everything happens per 0.2 s block, in logit space, followed by one decode:

1. **Team** (home vs away, given the ball is alive): the mean logit of the eight tabular team
   members (Ragha), plus `bilstm_soft` x the BiLSTM's log-odds of home against away (Tiya;
   team-swapped run), plus `bilstm_vote` towards the team named by each of her two decoders on
   those probabilities.
2. **Dead ball**: the mean logit of the two tabular dead-ball models, plus `live_gate` towards
   alive where the physics model v9d (Narmee) and the BiLSTM's own decoder (on the plain run)
   both say the ball is alive.
3. **Decode**: a 3-state Viterbi (home, away, dead) per half with switching costs and minimum
   lengths, then the kick-off rule (frames before each kick-off and kick-off set-ups are dead).
4. **Boundaries**: a possession that follows a stoppage starts `restart_delay` frames later; one
   that runs into a stoppage ends `stoppage_delay` frames later.

Output: the submission CSV, checked with the challenge's own validator.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

from main_pipeline.config import CHALLENGE, LETTERS, N_FRAMES, A, D, H, Paths, X
from models.bilstm.inference.bilstm_model import load_her_code
from models.tabular.inference.tabular_models import DEAD_BALL_MODELS, TEAM_MEMBERS, TEAM_MODELS


@dataclass(frozen=True)
class Settings:
    """The ensemble's settings: the weight of each model's evidence, the decoder's costs and
    minimum lengths, and the shifts of possession boundaries around stoppages.

    None was chosen on BAR-ATH data. Each follows from how the models were built, the rules of the
    game or the challenge's scoring, or is a model's own choice on open data. Evidence and costs
    share one unit (natural-log odds or probability), so they compare directly.

    Team (home vs away)
    - Eight tabular members, equal weights, averaged: a full {LightGBM, Kumo} x {columns b, ab}
      x {rule R2, E} design with nothing to rank one above another on BAR-ATH; averaging, not
      adding, counts their shared features and training data once. Together they are one voice.
    - bilstm_soft 1.5: the BiLSTM's log-odds of home against away is the logit of its P(home |
      alive) (the dead factor cancels), the tabular term's unit. As a second, independent family
      it gets at least parity; a little more because it reads 10 s or more of play either side of
      each frame, and under the challenge's convention possession is a matter of sequence (a loose
      ball stays with the last team in control); not much more because the tabular members
      learned the challenge's controlled-possession labels on two providers, the BiLSTM DFL's own
      ball-owning team on one.
    - bilstm_clip 8: odds of about 3,000 to 1. A softmax saturates far beyond that on single
      frames; the clip keeps one such frame from outweighing the rest of its block.
    - bilstm_vote 0.5: her decoded labels re-read probabilities bilstm_soft already counts, so they
      only break ties: +-0.5 each, +-1 when both agree (P 0.50 -> 0.73). Enough to settle an even
      block from the decoders' view of the whole run, never to overturn a confident one.
    - bilstm_decoders: Tiya's own, neither chosen on BAR-ATH: (4, 4, 4) is the decoder her
      checkpoint was selected with (PENALTY in train_twohead_v3.py), (1, 1, 6) the one she then
      chose on her development match (select_asymmetric_decoder.py).

    Dead ball
    - Two dead-ball models, equal weights, averaged, as for the team: they differ in how their
      training data imitates BAR-ATH's stoppages (SkillCorner's real ball, or a carried ball), and
      nothing favours one on BAR-ATH.
    - live_gate 1: v9d (rules on the tracking, Smart Tagging and the broadcast camera) and the
      BiLSTM are different kinds of model. Where both say the ball is in play, the block leans
      alive by up to one logit (P(alive) 0.50 -> 0.73, 0.20 -> 0.40): it tips blocks the dead-ball
      models are unsure of and cannot overturn a confident dead call. It only ever adds, so dead
      ball is still called only by the models built for BAR-ATH-like stoppages (and the kick-off
      rule).
    - dead_ball_window 12: the dead-ball models answer per 12-frame window; not a free choice.

    Decoder
    - costs (1, 2, 20), min_blocks (10, 1): the stoppage costs (2, 20) and the minimums (10, 1)
      are what leave-one-match-out on IDSSE chose for the tabular blends (RECIPES in
      models/tabular/src/lab/exp_k/bar_k2.py, grid in exp_sc/dur.py); for a change of team it
      chose 1 or 2 depending on the recipe.
      A stoppage inside a half is entered and left, so it costs 2 + 20 = 22 (the split matters
      only at the ends of a half). With one block's say capped at ln(1e4) = 9.2 by prob_floor,
      opening a stoppage in play takes at least 3 blocks (0.6 s) of decisive dead-ball evidence:
      a brief dip in P(alive) never becomes one. A change of team costs only 1 because the minimum
      length, not the cost, is what stops flicker; the lower of the two LOMO values suits team
      evidence already smoothed by the BiLSTM's long view.
      Possessions last at least 10 blocks (2 s) because flicker is the costlier error under
      spell F1: a fragment adds a false possession and usually cuts the true one around it below
      the IoU threshold, while missing a real possession under 2 s costs one. Stoppages need no
      minimum: their price of 22 already keeps them from being brief.
    - prob_floor and forced_p_alive 1e-4: as in Ragha's decoder (PROB_FLOOR in pipeline.py,
      FORCED_P_IN in exp_d/dball.py). Blocks the kick-off rule makes dead get the strongest dead
      evidence the floor allows: the rule is a law of the game (the ball is dead until the
      kick-off is taken), not an estimate.

    Boundaries
    - restart_delay 38 frames (1.5 s): the ball is dead until a restart is taken, but the end of a
      stoppage (the ball carried to the taker and placed, players back in shape) looks like play:
      exp_dball/carry.py shows a model trained on IDSSE's dead ball calling a carried or placed
      ball alive. The dead-ball models were built to reduce this; whatever remains starts
      possessions early. 1.5 s is the scale of that last phase in the team's hand-set restart
      model (carry.py: 0.4 s for a throw-in, 0.5-4 s of a placed ball for a set piece).
    - stoppage_delay 5 frames (0.2 s): no such bias is known where a possession ends, so the shift
      is the smallest there is, one block, towards the side spell F1 prefers to err on: a
      possession of length L ending d late keeps IoU L/(L+d), one ending d early (L-d)/L.
    - keep_frames 50 (2 s): the decoder's minimum possession, so a delay never makes a possession
      shorter than the decoder allows.
    """

    bilstm_soft: float = 1.5                    # weight of the BiLSTM's home/away log-odds
    bilstm_clip: float = 8.0                    # ... clipped to +- this
    bilstm_vote: float = 0.5                    # logit towards the team each of her decoders names
    bilstm_decoders: tuple = ((4.0, 4.0, 4.0),  # her decoders' costs: team switch, to dead, from dead
                              (1.0, 1.0, 6.0))  # (her 02 fixed and 03 asymmetric decoders)
    live_gate: float = 1.0                      # logit towards alive where v9d and her 03 say alive
    costs: tuple = (1.0, 2.0, 20.0)           # switching costs: change of team, into / out of a stoppage
    min_blocks: tuple = (10, 1)               # minimum length in blocks: possession (2 s), stoppage
    restart_delay: int = 38                   # frames (1.5 s)
    stoppage_delay: int = 5                   # frames (0.2 s)
    keep_frames: int = 50                     # a delayed possession keeps at least 2 s
    dead_ball_window: int = 12                # frames per dead-ball window
    prob_floor: float = 1e-4
    forced_p_alive: float = 1e-4              # P(alive) on blocks the kick-off rule makes dead


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Blocks:
    """The 0.2 s block layout of Ragha's team tables and the frame <-> block mappings."""

    def __init__(self, start: np.ndarray, end: np.ndarray, half: np.ndarray,
                 halves: list[tuple[int, int, int]]) -> None:
        self.start, self.end, self.half, self.halves = start, end, half, halves
        self.n = len(start)
        self.of_frame = np.full(N_FRAMES, -1, dtype=np.int64)
        for b, (a, e) in enumerate(zip(start, end)):
            self.of_frame[a:e] = b
        self.inside = self.of_frame >= 0
        self.count = np.bincount(self.of_frame[self.inside], minlength=self.n).astype(float)
        size = int(np.median(end - start))
        self.middle = np.minimum(start + size // 2, end - 1)
        self.first_frame_of_half = np.zeros(self.n, dtype=np.int64)
        for h, first, _ in halves:
            self.first_frame_of_half[half == h] = first

    def mean(self, per_frame: np.ndarray) -> np.ndarray:
        """Average of a per-frame series over each block."""
        sums = np.bincount(self.of_frame[self.inside], weights=per_frame[self.inside], minlength=self.n)
        return sums / np.maximum(self.count, 1)

    def window_of_middle(self, window: int) -> np.ndarray:
        """First frame of the dead-ball window holding each block's middle frame."""
        first = self.first_frame_of_half
        return first + ((self.middle - first) // window) * window


def viterbi_min(log_emission: np.ndarray, cost: np.ndarray, min_alive: int, min_dead: int) -> np.ndarray:
    """Best label path with switching costs and minimum run lengths.

    Each label is a chain of `minimum length` states that must be walked through before the
    label may change (the recursion of Ragha's decoder, ragha_tabular/lab/exp_sc/dur.py)."""
    length = {H: min_alive, A: min_alive, D: min_dead}
    first, s = {}, 0
    for c in (H, A, D):
        first[c] = s
        s += length[c]
    last = {c: first[c] + length[c] - 1 for c in (H, A, D)}
    label_of_state = np.empty(s, dtype=np.int8)
    for c in (H, A, D):
        label_of_state[first[c]:last[c] + 1] = c
    n = len(log_emission)
    score = np.full(s, -np.inf)
    for c in (H, A, D):
        score[first[c]:last[c] + 1] = log_emission[0, c]      # a half may open mid-run
    back = np.empty((n, s), dtype=np.int16)
    for t in range(1, n):
        new, pointer = np.empty(s), np.empty(s, dtype=np.int16)
        for c in (H, A, D):
            o, e = first[c], last[c]
            sources = [last[c2] for c2 in (H, A, D) if c2 != c]
            candidates = [score[last[c2]] - cost[c2, c] for c2 in (H, A, D) if c2 != c]
            if e == o:                                      # a 1-state label may also stay
                candidates.append(score[o])
                sources.append(o)
            k = int(np.argmax(candidates))
            new[o], pointer[o] = candidates[k], sources[k]
            if e > o:                                       # walk along the chain; the last state may stay
                new[o + 1:e + 1] = score[o:e]
                pointer[o + 1:e + 1] = np.arange(o, e, dtype=np.int16)
                if score[e] > score[e - 1]:
                    new[e], pointer[e] = score[e], e
            new[o:e + 1] += log_emission[t, c]
        score = new
        back[t] = pointer
    state = int(score.argmax())
    path = np.empty(n, dtype=np.int8)
    path[-1] = label_of_state[state]
    for t in range(n - 1, 0, -1):
        state = back[t, state]
        path[t - 1] = label_of_state[state]
    return path


class TeamEnsemble:
    """Reads every stage's outputs and builds the per-frame labels."""

    def __init__(self, paths: Paths, settings: Settings = Settings()) -> None:
        self.s = settings
        self.tabular = np.load(paths.tabular_outputs)
        decoder_inputs = np.load(paths.tabular_features / "decoder_inputs.npz")
        halves = [tuple(int(v) for v in row) for row in decoder_inputs["halves"]]
        self.blocks = Blocks(self.tabular["block_start"], self.tabular["block_end"],
                             self.tabular["block_half"], halves)
        self.forced = decoder_inputs["forced_dead"]
        bilstm = np.load(paths.bilstm_outputs)
        self.bilstm_raw = bilstm["raw"].astype(np.float64)
        self.bilstm_swapped = bilstm["swapped"].astype(np.float64)
        self.v9d = np.load(paths.physics_outputs)
        self.bilstm_code = load_her_code()

    def labels(self) -> np.ndarray:
        team, alive = self.team_logit(), self.alive_logit()
        path = self.decode(team, alive)
        codes = np.full(N_FRAMES, X, dtype=np.int8)
        codes[self.blocks.inside] = path[self.blocks.of_frame[self.blocks.inside]]
        codes = np.where(self.forced & (codes != X), D, codes).astype(np.int8)   # kick-off rule
        return self.delay_edges(codes)

    def team_logit(self) -> np.ndarray:
        team = np.mean([logit(self.tabular[f"team_{rule}_{member}"].astype(np.float64))
                        for rule in TEAM_MODELS for member in TEAM_MEMBERS], axis=0)
        for costs in self.s.bilstm_decoders:        # +1 per frame named home, -1 away, 0 dead
            lab = self._bilstm_decode(self.bilstm_swapped, costs)
            team = team + self.s.bilstm_vote * self.blocks.mean(np.where(lab == H, 1.0, np.where(lab == A, -1.0, 0.0)))
        p = self.bilstm_swapped
        log_odds = np.log(np.maximum(p[:, H], 1e-9)) - np.log(np.maximum(p[:, A], 1e-9))
        clipped = np.clip(log_odds, -self.s.bilstm_clip, self.s.bilstm_clip)
        return team + self.s.bilstm_soft * self.blocks.mean(clipped)

    def alive_logit(self) -> np.ndarray:
        window = self.blocks.window_of_middle(self.s.dead_ball_window)
        starts = self.tabular["window_start"].tolist()
        per_model = []
        for name in DEAD_BALL_MODELS:
            p_alive = dict(zip(starts, self.tabular[f"p_alive_{name}"].tolist()))
            per_model.append(logit(np.array([p_alive.get(int(k), 0.5) for k in window])))
        bilstm_03 = self._bilstm_decode(self.bilstm_raw, self.s.bilstm_decoders[1])
        both_alive = (self.v9d < D) & (bilstm_03 < D)
        return np.mean(per_model, axis=0) + self.s.live_gate * self.blocks.mean(both_alive.astype(float))

    def decode(self, team: np.ndarray, alive: np.ndarray) -> np.ndarray:
        s, blocks = self.s, self.blocks
        p_home = 1.0 / (1.0 + np.exp(-team))
        p_alive = np.where(self.forced[blocks.middle], s.forced_p_alive, 1.0 / (1.0 + np.exp(-alive)))
        emission = np.column_stack([p_alive * p_home, p_alive * (1 - p_home), 1 - p_alive])
        log_emission = np.log(np.clip(emission, s.prob_floor, 1.0))
        cost = np.zeros((3, 3))
        cost[H, A] = cost[A, H] = s.costs[0]
        cost[H, D] = cost[A, D] = s.costs[1]
        cost[D, H] = cost[D, A] = s.costs[2]
        path = np.empty(blocks.n, dtype=np.int8)
        for half in np.unique(blocks.half):
            selected = np.flatnonzero(blocks.half == half)
            path[selected] = viterbi_min(log_emission[selected], cost, *s.min_blocks)
        return path

    def delay_edges(self, codes: np.ndarray) -> np.ndarray:
        """Start a possession after a stoppage later (it keeps at least `keep_frames`) and end a
        possession before a stoppage later (the stoppage keeps at least one frame)."""
        s, out = self.s, codes.copy()
        for _, first, last in self.blocks.halves:
            w = codes[first:last + 1]
            bounds = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
            runs = [(int(bounds[i]), int(bounds[i + 1]), int(w[bounds[i]])) for i in range(len(bounds) - 1)]
            o = out[first:last + 1]
            for i, (a, b, label) in enumerate(runs):
                if label not in (H, A):
                    continue
                if i and runs[i - 1][2] == D:
                    o[a:a + min(s.restart_delay, max(0, (b - a) - s.keep_frames))] = D
                if i + 1 < len(runs) and runs[i + 1][2] == D:
                    o[b:b + min(s.stoppage_delay, max(0, (runs[i + 1][1] - runs[i + 1][0]) - 1))] = label
        return out

    def _bilstm_decode(self, probabilities: np.ndarray, costs: tuple) -> np.ndarray:
        """Tiya's own Viterbi decoder over the whole match as one span."""
        return self.bilstm_code.asymmetric_decode(probabilities, [(0, N_FRAMES)], *costs)


def write_submission(codes: np.ndarray, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"{frame},{LETTERS[code]}" for frame, code in enumerate(codes))
    path.write_text("frame,label\n" + rows + "\n")


def main() -> None:
    paths = Paths.from_env()
    codes = TeamEnsemble(paths).labels()
    write_submission(codes, paths.submission)
    subprocess.run([sys.executable, str(CHALLENGE / "evaluation" / "validate_submission.py"),
                    str(paths.submission)], check=True)
    graded = codes != X
    summary = {"rows": int(len(codes)), "dead_ball_share": round(float(np.mean(codes[graded] == D)), 4)}
    print(f"ensemble: wrote {paths.submission} {json.dumps(summary)}")


if __name__ == "__main__":
    main()
