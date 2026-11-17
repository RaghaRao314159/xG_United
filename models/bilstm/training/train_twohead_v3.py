"""Train schema-3 LSTM or TCN candidates on rich shared tracking inputs."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
from prepare_model_inputs_v3 import FEATURES, LENGTH, WIDTH

LABELS = ['H', 'A', 'D']
TRAIN_BASE = ('J03WPY', 'J03WMX', 'J03WOH', 'J03WQQ')
TRAIN = tuple(name for match in TRAIN_BASE for name in (match, match + '-player-gaps'))
DEV = 'J03WN1'
RESERVED = 'J03WR9'
CORE, HALO = 512, 256
HIDDEN_BALL = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
PENALTY = 4


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def segments(df):
    starts = np.flatnonzero((df.period_id.ne(df.period_id.shift()) |
                            df.frame_id.diff().ne(1)).to_numpy())
    return list(zip(starts.tolist(), np.r_[starts[1:], len(df)].tolist()))


def plan(spans, core=CORE):
    return np.array([(lo, hi, start) for lo, hi in spans
                     for start in range(lo, hi, core)], dtype=np.int64)


def indices_for(windows, core=CORE, halo=HALO):
    indices = windows[:, 2, None] + np.arange(-halo, core + halo)[None, :]
    indices = np.clip(indices, windows[:, 0, None], windows[:, 1, None] - 1)
    valid = windows[:, 2, None] + np.arange(core)[None, :] < windows[:, 1, None]
    return indices, valid


def hide_ball(raw, mask):
    """Erase every ball-dependent input, including derivatives after a hidden frame."""
    out = raw.copy()
    for column in HIDDEN_BALL:
        out[..., column] = np.where(mask, np.nan, out[..., column])
    out[..., 5] = np.where(mask, 0., out[..., 5])
    velocity_missing = mask.copy()
    velocity_missing[..., 1:] |= mask[..., :-1]
    out[..., 2:5] = np.where(velocity_missing[..., None], np.nan, out[..., 2:5])
    return out


def augment(raw, targets, rng):
    raw, targets = raw.copy(), targets.copy()
    batch, length, _ = raw.shape
    flip = rng.random(batch) < .5
    for column in (0, 13, 22):
        raw[flip, :, column] = LENGTH - raw[flip, :, column]
    for column in (2, 17, 26):
        raw[flip, :, column] *= -1
    flip = rng.random(batch) < .5
    for column in (1, 14, 23):
        raw[flip, :, column] = WIDTH - raw[flip, :, column]
    for column in (3, 18, 27):
        raw[flip, :, column] *= -1
    swap = rng.random(batch) < .5
    distances = raw[swap, :, 6:12].copy()
    raw[swap, :, 6:9], raw[swap, :, 9:12] = distances[..., 3:6], distances[..., :3]
    teams = raw[swap, :, 12:30].copy()
    raw[swap, :, 12:21], raw[swap, :, 21:30] = teams[..., 9:18], teams[..., :9]
    targets[swap] = np.where((targets[swap] >= 0) & (targets[swap] < 2),
                             1 - targets[swap], targets[swap])
    mask = np.zeros((batch, length), dtype=bool)
    for sample in range(batch):
        for _ in range(int(rng.integers(0, 13))):
            start = int(rng.integers(length))
            end = min(length, start + int(rng.integers(13, 126)))
            mask[sample, start:end] = True
    return hide_ball(raw, mask), targets


def stress_mask(n, spans, seed=2026):
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    for lo, hi in spans:
        target = int((hi - lo) * .4)
        covered = 0
        while covered < target:
            start = int(rng.integers(lo, hi))
            end = min(hi, start + int(rng.integers(13, 126)))
            covered += int((~mask[start:end]).sum())
            mask[start:end] = True
    return mask


def normalize(values, mean, scale):
    missing = np.isnan(values)
    normalized = np.clip(np.nan_to_num((values - mean) / scale), -20, 20)
    return np.concatenate([normalized, missing.astype(np.float32)], axis=-1).astype(np.float32)


def make_model(torch, hidden=256, kind='lstm'):
    class PossessionLSTM(torch.nn.Module):
        def __init__(self):
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

    class PossessionTCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(len(FEATURES) * 2, hidden)
            self.blocks = torch.nn.ModuleList([
                torch.nn.Conv1d(hidden, hidden, 3, dilation=2**i, padding=2**i)
                for i in range(8)])
            self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden) for _ in range(8)])
            self.dropout = torch.nn.Dropout(.15)
            self.dead_head = torch.nn.Sequential(torch.nn.Linear(hidden, 128),
                                                  torch.nn.GELU(), torch.nn.Linear(128, 1))
            self.team_head = torch.nn.Sequential(torch.nn.Linear(hidden, 128),
                                                  torch.nn.GELU(), torch.nn.Linear(128, 2))

        def forward(self, x):
            x = self.encoder(x).transpose(1, 2)
            for convolution, norm in zip(self.blocks, self.norms):
                residual = norm(convolution(x).transpose(1, 2))
                residual = self.dropout(torch.nn.functional.gelu(residual)).transpose(1, 2)
                x = x + residual
            sequence = x.transpose(1, 2)
            return self.dead_head(sequence), self.team_head(sequence)

    if kind == 'lstm':
        return PossessionLSTM()
    if kind == 'tcn':
        return PossessionTCN()
    raise ValueError('Unknown model architecture')


def probabilities_from_heads(torch, dead_logits, team_logits):
    """Combine P(dead) and P(home/away | alive) into H/A/D probabilities."""
    dead = torch.sigmoid(dead_logits)
    teams = torch.softmax(team_logits, dim=-1)
    alive = 1.0 - dead
    return torch.cat((alive * teams, dead), dim=-1)


def core_self_checks():
    df = pd.DataFrame({'period_id': [1, 1, 2, 2, 2], 'frame_id': [1, 2, 5, 6, 9]})
    assert segments(df) == [(0, 2), (2, 4), (4, 5)]
    windows = plan(segments(df), core=3)
    indices, valid = indices_for(windows, core=3, halo=2)
    assert valid.sum() == len(df)
    for window, row in zip(windows, indices):
        assert (row >= window[0]).all() and (row < window[1]).all()
    raw = np.ones((1, 5, len(FEATURES)), dtype=np.float32)
    raw[..., 5] = 1
    original = raw.copy()
    mask = np.array([[False, True, True, False, False]])
    hidden = hide_ball(raw, mask)
    assert np.array_equal(raw, original)
    assert np.isnan(hidden[0, 1:3, HIDDEN_BALL]).all()
    assert np.isnan(hidden[0, 3, 2:5]).all()
    assert np.isfinite(hidden[0, 3, 0]) and hidden[0, 3, 5] == 1
    assert (hidden[0, 1:3, 5] == 0).all()
    np.testing.assert_array_equal(hidden[..., 12:], raw[..., 12:])
    result = normalize(hidden, np.zeros(len(FEATURES)), np.ones(len(FEATURES)))
    assert np.isfinite(result).all() and result.shape[-1] == len(FEATURES) * 2
    assert (result[0, 1:3, 0] == 0).all() and (result[0, 1:3, len(FEATURES)] == 1).all()
    stressed = stress_mask(200, [(0, 100), (100, 200)])
    assert stressed[:100].sum() >= 40 and stressed[100:].sum() >= 40
    assert np.array_equal(stressed, stress_mask(200, [(0, 100), (100, 200)]))
    y = np.array([[0, 1, 2, -100, 0]])
    a, b = augment(raw, y, np.random.default_rng(42))
    assert np.array_equal(y, [[0, 1, 2, -100, 0]])
    assert b[0, 2] == 2 and b[0, 3] == -100
    assert set(FEATURES).isdisjoint({'ball_z', 'ball_speed', 'label', 'ball_state', 'ball_owning_team_id'})


def load_match(input_root, match):
    folder = input_root / match
    summary = json.loads((folder / 'summary.json').read_text())
    assert summary['complete'] and summary['schema'] == 3 and summary['source'] == 'idsse'
    assert summary['features'] == FEATURES and summary['fps'] == 25
    assert summary['reference_pitch_metres'] == [LENGTH, WIDTH]
    assert not summary['height_used'] and not summary['provider_speed_used']
    df = pd.read_csv(folder / 'inputs.csv.gz')
    targets = pd.read_csv(folder / 'targets.csv.gz')
    assert len(df) == len(targets) == summary['frames']
    assert np.array_equal(df.frame, np.arange(len(df)))
    assert np.array_equal(df.frame, targets.frame)
    assert not df.duplicated(['period_id', 'frame_id']).any()
    x = df[FEATURES].to_numpy(dtype=np.float32)
    assert not np.isinf(x).any()
    assert np.isin(x[:, 5], [0, 1]).all()
    assert np.array_equal(x[:, 5].astype(bool), np.isfinite(x[:, :2]).all(axis=1))
    y = targets.label.map({'H': 0, 'A': 1, 'D': 2})
    assert y.notna().all()
    return df[['frame', 'period_id', 'frame_id']], x, y.to_numpy(dtype=np.int64), segments(df)


def predict(torch, model, values, spans, mean, scale, batch_size):
    probabilities = np.full((len(values), 3), np.nan, dtype=np.float32)
    windows = plan(spans)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            chosen = windows[start:start + batch_size]
            indices, _ = indices_for(chosen)
            x = torch.from_numpy(normalize(values[indices], mean, scale)).cuda()
            dead_logits, team_logits = model(x)
            result = probabilities_from_heads(
                torch, dead_logits[:, HALO:HALO + CORE], team_logits[:, HALO:HALO + CORE]
            ).cpu().numpy()
            for row, (_, hi, centre) in enumerate(chosen):
                count = min(CORE, hi - centre)
                probabilities[centre:centre + count] = result[row, :count]
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1., atol=1e-5)
    return probabilities


def sequence_metrics(pm, decoder, truth, probabilities, spans):
    raw = probabilities.argmax(axis=1)
    decoded = np.empty(len(truth), dtype=np.int8)
    emissions = np.log(np.maximum(probabilities, 1e-12)).tolist()
    for lo, hi in spans:
        decoded[lo:hi] = decoder.decode_segment(emissions[lo:hi], PENALTY)
    boundaries = [lo for lo, _ in spans[1:]]
    gt = np.insert(truth, boundaries, pm.X)
    report = {}
    for name, prediction in [('raw', raw), ('decoded', decoded)]:
        score = pm.evaluate(gt, np.insert(prediction, boundaries, pm.X))
        report[name] = {key: score[key] for key in (
            'macro_f1', 'spell_f1_mean', 'n_gt_spells', 'n_pred_spells')}
    return report, decoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=Path.home() /
                        'proj/tiya/more-than-a-hack-2027/data/model-inputs-v3')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--model', choices=('lstm', 'tcn'), default='lstm')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    assert args.epochs > 0 and args.batch_size > 0
    core_self_checks()
    if args.self_test and importlib.util.find_spec('torch') is None:
        print('Core checks passed; PyTorch is unavailable here, so neural checks were NOT run.')
        return
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    check_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tiny = make_model(torch, hidden=8, kind=args.model).to(check_device)
    example = torch.zeros(2, 16, len(FEATURES) * 2, device=check_device)
    dead_scores, team_scores = tiny(example)
    assert dead_scores.shape == (2, 16, 1) and team_scores.shape == (2, 16, 2)
    loss = (torch.nn.functional.binary_cross_entropy_with_logits(
        dead_scores.squeeze(-1), torch.zeros((2, 16), device=check_device)) +
        torch.nn.functional.cross_entropy(team_scores.reshape(-1, 2),
                                          torch.zeros(32, dtype=torch.long, device=check_device)))
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in tiny.parameters())
    del tiny, dead_scores, team_scores, loss
    if args.self_test:
        print(f'Core and neural forward/backward self-checks passed on {check_device}.')
        return
    assert torch.cuda.is_available(), 'CUDA is unavailable'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    project = args.input_root.parent.parent
    sys.path.insert(0, str(project / 'evaluation'))
    import possession_metrics as pm
    import possession_decode as decoder
    decoder.self_check()
    run_root = args.run_root or project / 'data/runs'
    out = run_root / datetime.now(timezone.utc).strftime(f'{args.model}-v3-seed{args.seed}-%Y%m%d-%H%M%S')
    out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(out / 'train.log'), logging.StreamHandler(sys.stdout)])
    log = logging.getLogger(__name__)
    log.info('Output: %s', out)
    log.info('GPU: %s; PyTorch: %s', torch.cuda.get_device_name(0), torch.__version__)
    log.info('Train: %s; development: %s; reserved and NOT loaded: %s', TRAIN, DEV, RESERVED)
    log.info('Context %.2fs; fixed decoding penalty %s; synthetic gaps are a stress test, not calibrated Metrica noise.',
             (CORE + HALO * 2) / 25, PENALTY)
    config = {'train': TRAIN, 'development': DEV, 'reserved': RESERVED,
              'input_root': str(args.input_root.resolve()), 'project_root': str(project.resolve()),
              'features': FEATURES, 'core': CORE, 'halo': HALO, 'hidden': 256, 'model': args.model,
              'heads': {'dead': 'binary', 'team_given_alive': 'binary'},
              'decoder_penalty': PENALTY, 'seed': args.seed, 'epochs': args.epochs,
              'batch_size': args.batch_size, 'torch': str(torch.__version__),
              'numpy': np.__version__, 'input_hashes': {}}
    for match in (*TRAIN, DEV, DEV + '-player-gaps'):
        for filename in ('inputs.csv.gz', 'targets.csv.gz', 'summary.json'):
            p = args.input_root / match / filename
            config['input_hashes'][f'{match}/{filename}'] = file_hash(p)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    for path in (Path(__file__), Path(decoder.__file__), Path(sys.modules['prepare_model_inputs_v3'].__file__),
                 Path(sys.modules['prepare_model_inputs'].__file__)):
        shutil.copyfile(path, out / path.name)
    arrays, labels, train_spans = [], [], []
    offset = 0
    for match in TRAIN:
        _, x, y, spans = load_match(args.input_root, match)
        log.info('Loaded training %s: %s frames', match, len(x))
        arrays.append(x)
        labels.append(y)
        train_spans.extend((lo + offset, hi + offset) for lo, hi in spans)
        offset += len(x)
    train_x, train_y = np.concatenate(arrays), np.concatenate(labels)
    del arrays, labels
    dev_df, dev_x, dev_y, dev_spans = load_match(args.input_root, DEV)
    mean = np.nanmean(train_x, axis=0)
    scale = np.nanstd(train_x, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.)
    # A constant visible flag is still meaningful when augmentation changes it to zero.
    scale[[5, 12, 21]] = np.maximum(scale[[5, 12, 21]], 1.)
    assert np.isfinite(mean).all() and np.isfinite(scale).all()
    dev_mask = stress_mask(len(dev_x), dev_spans)
    stress_df, player_stress, stress_y, stress_spans = load_match(args.input_root, DEV + '-player-gaps')
    assert stress_df.equals(dev_df) and np.array_equal(stress_y, dev_y) and stress_spans == dev_spans
    dev_stress = hide_ball(player_stress, dev_mask)
    log.info('Synthetic development ball gaps: %.1f%%, plus persistent player-track gaps', 100 * dev_mask.mean())
    model = make_model(torch, kind=args.model).cuda()
    log.info('Parameters: %s; training windows: %s', sum(p.numel() for p in model.parameters()), len(plan(train_spans)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    frequencies = np.bincount(train_y, minlength=3)
    assert (frequencies > 0).all()
    dead_loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(frequencies[:2].sum() / frequencies[2], dtype=torch.float32, device='cuda'))
    team_weights = np.sqrt(frequencies[:2].sum() / (2 * frequencies[:2]))
    team_loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(team_weights, dtype=torch.float32, device='cuda'))
    rng = np.random.default_rng(args.seed)
    windows = plan(train_spans)
    best, stale = -1., 0
    history = []
    try:
        for epoch in range(1, args.epochs + 1):
            started = time.monotonic()
            model.train()
            order = rng.permutation(len(windows))
            total_loss, total_frames = 0., 0
            for start in range(0, len(order), args.batch_size):
                chosen = windows[order[start:start + args.batch_size]]
                indices, valid = indices_for(chosen)
                target = train_y[indices[:, HALO:HALO + CORE]].copy()
                target[~valid] = -100
                raw, target = augment(train_x[indices], target, rng)
                x = torch.from_numpy(normalize(raw, mean, scale)).cuda()
                target = torch.from_numpy(target).cuda()
                optimizer.zero_grad(set_to_none=True)
                dead_logits, team_logits = model(x)
                dead_logits = dead_logits[:, HALO:HALO + CORE, 0]
                team_logits = team_logits[:, HALO:HALO + CORE]
                valid_target = target >= 0
                dead_loss = dead_loss_fn(dead_logits[valid_target], (target[valid_target] == 2).float())
                team_target = target < 2
                # A window can be entirely dead-ball. Its team head has no valid target,
                # and CrossEntropyLoss over an empty tensor returns NaN.
                team_loss = (team_loss_fn(team_logits[team_target], target[team_target])
                             if team_target.any() else dead_logits.sum() * 0.)
                loss = dead_loss + team_loss
                assert torch.isfinite(loss), 'Non-finite training loss'
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                count = int(valid.sum())
                total_loss += loss.item() * count
                total_frames += count
            scheduler.step()
            log.info('Epoch %s/%s: loss %.4f; training %.1fs; peak GPU %.2f GiB',
                     epoch, args.epochs, total_loss / total_frames, time.monotonic() - started,
                     torch.cuda.max_memory_allocated() / 2**30)
            if epoch != 1 and epoch % 2 and epoch != args.epochs:
                continue
            validation = {}
            outputs = {}
            for name, values in [('clean', dev_x), ('synthetic_gaps', dev_stress)]:
                log.info('Evaluating development: %s...', name)
                probabilities = predict(torch, model, values, dev_spans, mean, scale, args.batch_size)
                report, decoded = sequence_metrics(pm, decoder, dev_y, probabilities, dev_spans)
                validation[name] = report
                outputs[name] = (probabilities, decoded)
                log.info('%s: raw macroF1 %.4f, raw spellF1 %.4f; decoded spellF1 %.4f; possessions %s/%s',
                         name, report['raw']['macro_f1'], report['raw']['spell_f1_mean'],
                         report['decoded']['spell_f1_mean'], report['decoded']['n_pred_spells'],
                         report['decoded']['n_gt_spells'])
            selection_score = float(np.mean([v['decoded']['spell_f1_mean'] for v in validation.values()]))
            record = {'epoch': epoch, 'train_loss': total_loss / total_frames,
                      'selection_score': selection_score, 'development': validation}
            history.append(record)
            (out / 'history.json').write_text(json.dumps(history, indent=2))
            if selection_score > best:
                best, stale = selection_score, 0
                checkpoint = {'state_dict': model.state_dict(), 'config': config,
                              'mean': mean.tolist(), 'scale': scale.tolist(),
                              'epoch': epoch, 'development': validation, 'selection_score': best}
                temporary = out / 'best.pt.tmp'
                torch.save(checkpoint, temporary)
                temporary.replace(out / 'best.pt')
                for name, (probabilities, decoded) in outputs.items():
                    frame = dev_df.copy()
                    frame['truth'] = np.array(LABELS)[dev_y]
                    frame['raw_prediction'] = np.array(LABELS)[probabilities.argmax(axis=1)]
                    frame['decoded_prediction'] = np.array(LABELS)[decoded]
                    frame[[f'prob_{label}' for label in LABELS]] = probabilities
                    frame.to_csv(out / f'dev_{name}_predictions.csv.gz', index=False)
                log.info('Saved new best checkpoint: epoch %s, selection score %.4f', epoch, best)
            else:
                stale += 1
            if stale >= 6:
                log.info('Stopping after six development evaluations without improvement.')
                break
    except KeyboardInterrupt:
        log.warning('Stopped by user. Any best.pt already saved remains available.')
        return
    (out / 'completed.json').write_text(json.dumps({'best_selection_score': best, 'epochs_completed': epoch}))
    log.info('Finished. Best development selection score %.4f; checkpoint %s', best, out / 'best.pt')
    log.info('No final-test match or competition labels were loaded. Development scores are not final results.')


if __name__ == '__main__':
    main()
