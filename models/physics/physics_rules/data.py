"""Load any match (the BAR-ATH challenge match or any converted IDSSE / SkillCorner match)
into numpy arrays in metres and video frames, cached on disk.
"""

import io
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import (DATA_DIR, GAME, GAMES, anchor_video_frame, asset_paths, half_video_blocks,
                   require, source_frames)  # noqa: E402

CACHE_DIR = os.path.join(DATA_DIR, "cache")
CONV_DIR = os.path.join(DATA_DIR, "external", "converted")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
HOME, AWAY = 0, 1


class Match(object):
    """Everything the methods need, in metres and video frames.

    ball      (n, 2) float  ball position in metres, NaN where unseen
    ball_seen (n,)   bool
    px, py    (n, S) float  player positions in metres per track slot, NaN where unseen
    pspeed    (n, S) float  player speed m/s as reported by the tracker
    team      (S,)   int8   0 home / 1 away per slot
    halves    [(half, first_frame, last_frame)] inclusive video frames
    tags      list of Smart Tagging dicts (see starter/load_smart_tagging.py)
    """

    def __init__(self, ball, px, py, pspeed, team, team_names, tags, game=GAME,
                 fps=None, pitch=None, halves=None, gt_path=None, source_repo="Metrica Challenge",
                 league="LaLiga", title=None):
        self.ball = ball
        self.ball_seen = ~np.isnan(ball[:, 0])
        self.px, self.py, self.pspeed = px, py, pspeed
        self.team = team
        self.team_names = team_names
        self.tags = tags
        self.game = game
        self.n = len(ball)
        self.fps = fps or GAMES[game]["fps"]
        self.length, self.width = pitch or GAMES[game]["pitch"]
        self.halves = halves or half_video_blocks(game)
        self.gt_path = gt_path or PUBLIC_GT
        self.source_repo = source_repo
        self.league = league
        self.title = title or f"{team_names[0]} vs {team_names[1]}"
        self.is_challenge = (game == GAME)

    def graded_mask(self):
        mask = np.zeros(self.n, dtype=bool)
        for _, first, last in self.halves:
            mask[first:last + 1] = True
        return mask


def _team_of_slots(meta_path):
    root = ET.parse(meta_path).getroot()
    teams = [(t.get("id"), t.findtext("Name")) for t in root.iter("Team")]
    home_id = teams[0][0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    team = np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)
    return team, (teams[0][1], teams[1][1])


def _parse_raw(raw_path, n_frames, n_slots):
    """Raw EPTS text -> (players (n, S, 3), ball (n, 3)). One flat CSV read."""
    with open(raw_path) as handle:
        text = handle.read()
    text = text.replace(":", ",").replace(";", ",")
    n_cols = 1 + 3 * n_slots + 3
    table = pd.read_csv(io.StringIO(text), header=None, names=range(n_cols),
                        dtype=np.float64, na_values=["NaN"], engine="c")
    del text
    values = table.values
    frame_ids = values[:, 0].astype(np.int64)
    players = values[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball = values[:, 1 + 3 * n_slots:1 + 3 * n_slots + 3]

    out_p = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    out_b = np.full((n_frames, 3), np.nan, dtype=np.float32)
    index = frame_ids - 1
    ok = (index >= 0) & (index < n_frames)
    out_p[index[ok]] = players[ok]
    out_b[index[ok]] = ball[ok]
    return out_p, out_b


def _load_smart_tagging_xml(xml_path, fps):
    out = []
    if not os.path.isfile(xml_path):
        return out
    for instance in ET.parse(xml_path).iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        out.append({
            "code": instance.findtext("code"),
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
            "tags": {label.findtext("group"): label.findtext("text")
                     for label in instance.findall("label")},
        })
    return out


def load_smart_tagging(game=GAME):
    if game in GAMES:
        import load_smart_tagging as lst
        return lst.load(game)
    cdir = os.path.join(CONV_DIR, game)
    with open(os.path.join(cdir, f"{game}_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    xml_path = os.path.join(cdir, "tactical", f"{game}_tactical_smart_tagging.xml")
    return _load_smart_tagging_xml(xml_path, meta["fps"])


def list_games(include_external=True):
    """Return ordered list of game codes: ['BAR-ATH', 'IDSSE-J03WMX', ..., 'SC-1874553', ...]."""
    games = [GAME]
    if include_external and os.path.isdir(CONV_DIR):
        ext = sorted(d for d in os.listdir(CONV_DIR)
                     if os.path.isfile(os.path.join(CONV_DIR, d, f"{d}_meta.json")))
        games.extend(ext)
    return games


def load_match(game=GAME, refresh=False, verbose=True):
    if game in GAMES:
        meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
        team, team_names = _team_of_slots(meta_path)
        n_frames = source_frames("tactical", game)
        length, width = GAMES[game]["pitch"]
        fps = GAMES[game]["fps"]
        halves = half_video_blocks(game)
        gt_path = PUBLIC_GT
        source_repo = "Metrica Challenge"
        league = "LaLiga EA Sports"
        title = f"{team_names[0]} vs {team_names[1]}"
        tags = load_smart_tagging(game)
    else:
        cdir = os.path.join(CONV_DIR, game)
        with open(os.path.join(cdir, f"{game}_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta_path = os.path.join(cdir, "tactical", f"{game}_tactical_atd.xml")
        raw_path = os.path.join(cdir, "tactical", f"{game}_tactical_atd.txt")
        st_path = os.path.join(cdir, "tactical", f"{game}_tactical_smart_tagging.xml")
        team, team_names = _team_of_slots(meta_path)
        n_frames = int(meta["n_video_frames"])
        length, width = float(meta["pitch"][0]), float(meta["pitch"][1])
        fps = int(meta["fps"])
        halves = [(int(h), int(a), int(b)) for h, a, b in meta["halves"]]
        gt_path = os.path.join(cdir, f"{game}_possession_gt.csv")
        source_repo = meta.get("source_repo", "External")
        league = meta.get("league", "")
        title = meta.get("title", f"{team_names[0]} vs {team_names[1]}")
        tags = _load_smart_tagging_xml(st_path, fps)

    cache = os.path.join(CACHE_DIR, f"{game}_tactical_atd.npz")
    if os.path.isfile(cache) and not refresh:
        blob = np.load(cache)
        players, ball = blob["players"], blob["ball"]
    else:
        if verbose:
            print(f"parsing {os.path.basename(raw_path)} (first run only) ...")
        players, ball = _parse_raw(raw_path, n_frames, len(team))
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache, players=players, ball=ball)
        if verbose:
            print(f"  cached -> {os.path.relpath(cache, ROOT)}")

    ball_m = np.stack([ball[:, 0] * length, ball[:, 1] * width], axis=1).astype(np.float64)
    px = players[:, :, 0].astype(np.float64) * length
    py = players[:, :, 1].astype(np.float64) * width
    pspeed = players[:, :, 2].astype(np.float64)
    return Match(ball_m, px, py, pspeed, team, team_names, tags, game=game,
                 fps=fps, pitch=(length, width), halves=halves, gt_path=gt_path,
                 source_repo=source_repo, league=league, title=title)
