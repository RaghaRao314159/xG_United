"""Build the submission file from the provided match data.

    python -m main_pipeline                      # every stage, in order, on a CUDA GPU if there is one
    python -m main_pipeline --device cpu         # CPU only (slow: hours, for the Kumo models)
    python -m main_pipeline --data /path/to/data # the provided match data (default: ./data)
    python -m main_pipeline --stages ensemble    # only some stages (their inputs must exist)

The stages that run a model live with that model, in models/<model>/inference/; this package
holds the runner, the shared configuration, the weight check and the ensemble. Each stage runs in
its own Python process: the models' code keeps settings in module-level globals, so separate
processes keep the stages independent of one another.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .config import DEFAULT_THREADS, REPO, Paths

STAGES = {   # stage -> (module run as `python -m <module>`, what it does)
    "check_weights": ("main_pipeline.check_weights",
                      "check every weight file against weights.lock.json; download NVIDIA's Kumo network"),
    "tabular_features": ("models.tabular.inference.tabular_features",
                         "features of the tabular models from the tactical tracking (Ragha)"),
    "tabular_models": ("models.tabular.inference.tabular_models",
                       "8 tabular team members and 2 dead-ball models (Ragha)"),
    "bilstm_model": ("models.bilstm.inference.bilstm_model", "two-head BiLSTM (Tiya)"),
    "physics_model": ("models.physics.inference.physics_model", "physics model v9d (Narmee)"),
    "ensemble": ("main_pipeline.ensemble", "combine, decode and write the submission"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m main_pipeline", description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=REPO / "data",
                        help="the provided match data (tactical/ and broadcast/ folders)")
    parser.add_argument("--work", type=Path, default=REPO / "work", help="intermediate outputs")
    parser.add_argument("--out", type=Path, default=REPO / "submission")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="for Kumo and the BiLSTM; auto = CUDA if available, else CPU")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="CPU threads")
    parser.add_argument("--verify-reference", action="store_true",
                        help="require the output to match the submitted CSV's SHA-256")
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(data=args.data.resolve(), work=args.work.resolve(), out=args.out.resolve())
    for name in (stage for stage in STAGES if stage in args.stages):    # always in pipeline order
        module, what = STAGES[name]
        print(f"== {name}: {what}", flush=True)
        started = time.time()
        env = paths.environment(args.threads, args.device, offline=(name != "check_weights"))
        subprocess.run([sys.executable, "-m", module], env=env, cwd=REPO, check=True)
        print(f"   {name} took {time.time() - started:.0f} s", flush=True)
    if args.verify_reference:
        from .verify_submission import verify_reference
        verify_reference(paths.submission)
    print(f"submission: {paths.submission}")


if __name__ == "__main__":
    main()
