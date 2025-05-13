# -*- coding: utf-8 -*-
"""
Created on Mon May 12 00:39:30 2025 — revised May 12 2025

@author: ADMIN

Score‑fusion utilities for spoof‑detection systems.

Supported fusion techniques
===========================
1. **MEAN**  – Simple arithmetic mean of scores (no training)
2. **PSO**   – Particle Swarm Optimisation over the weight simplex (EER‑driven)
3. **LOGREG** – Logistic‑regression calibration / linear fusion
4. **DE**    – Differential Evolution (SciPy) on the weight simplex

**NEW**: The script now **saves the fused scores** in the *same utterance order*
as the first input score file. Use the ``--out-file`` flag to set the path
(default: ``fused_scores.txt``).

Quick examples
--------------
```bash
# Mean fusion, write to default fused_scores.txt
python score_fusion.py

# PSO fusion and save to custom path
python score_fusion.py --method pso --out-file results/pso_fused.txt \
                       scores/sys1.txt scores/sys2.txt
```

Output file format matches the original (``utt_id tag label score``), with the
*tag* copied from the first input file and *score* replaced by the fused score.

Dependencies
------------
* Mandatory: ``numpy`` ≥1.20, your local ``eval_metrics.compute_eer`` function.
* Optional  : ``scipy`` ≥1.7  (*required* for *logreg* with SciPy and for *de*).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from eval_metrics import compute_eer  # project‑local
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf

###############################################################################
# Configuration
###############################################################################

DEFAULT_SCORE_FILES: list[str] = [
    "models/hubertBase_ecapa_tdnn/eval_scores_with_tags.txt",
    "models/hubertLarge_ecapa_tdnn/eval_scores_with_tags.txt",
]
DEFAULT_OUT_PATH = "models/fused_scoresBase_Large-pso.txt"

###############################################################################
# File I/O
###############################################################################

def load_scores(score_file: str | Path) -> dict[str, Tuple[float, int]]:
    """Load ASVspoof‑format score file: ``utt_id tag label score`` (tag ignored)."""
    scores: dict[str, Tuple[float, int]] = {}
    with Path(score_file).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            utt_id, _tag, label_str, score_str = line.strip().split()
            label = 0 if label_str.lower() == "bonafide" else 1
            scores[utt_id] = (float(score_str), label)
    return scores


def write_fused_scores(template_score_file: str | Path, fused_dict: dict[str, Tuple[float, int]], out_path: str | Path) -> None:
    """Write a fused‑score file preserving original utterance ordering & tags."""
    out_path = Path(out_path)
    with Path(template_score_file).expanduser().open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            utt_id, tag, label_str, _orig_score = line.strip().split()
            fused_score = fused_dict[utt_id][0]
            fout.write(f"{utt_id} {tag} {label_str} {fused_score:.6f}\n")

###############################################################################
# Generic helpers
###############################################################################
def pso_optimize_weights(
    score_dicts: List[dict],
    num_particles: int = 30,
    max_iter: int = 150,
    w_inertia: float = 0.729,
    c1: float = 1.49445,
    c2: float = 1.49445,
    seed: int | None = 42,
) -> np.ndarray:
    """Search the simplex for EER‑optimal fusion weights using PSO."""

    n_systems = len(score_dicts)
    rng = np.random.default_rng(seed)

    # Objective: minimise EER
    def _objective(weights: np.ndarray) -> float:
        fused = fuse_scores_weighted(score_dicts, weights)
        return compute_eer_from_scores(fused)

    # Initialise swarm in the simplex
    positions = rng.dirichlet(np.ones(n_systems), size=num_particles)
    velocities = np.zeros_like(positions)

    pbest_pos = positions.copy()
    pbest_err = np.array([_objective(p) for p in positions])
    gbest_idx = int(np.argmin(pbest_err))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_err = pbest_err[gbest_idx]

    # Iterate
    for _ in range(max_iter):
        for i in range(num_particles):
            r1 = rng.random(n_systems)
            r2 = rng.random(n_systems)
            velocities[i] = (
                w_inertia * velocities[i]
                + c1 * r1 * (pbest_pos[i] - positions[i])
                + c2 * r2 * (gbest_pos - positions[i])
            )
            positions[i] += velocities[i]

            # Project back to simplex (positive, sums to 1)
            positions[i] = np.clip(positions[i], 1e-9, None)
            positions[i] /= positions[i].sum()

            # Evaluate
            err = _objective(positions[i])
            if err < pbest_err[i]:
                pbest_pos[i] = positions[i].copy()
                pbest_err[i] = err
                if err < gbest_err:
                    gbest_pos = positions[i].copy()
                    gbest_err = err

    return gbest_pos

def compute_eer_from_scores(score_dict: dict[str, Tuple[float, int]]) -> float:
    bona = [s for s, l in score_dict.values() if l == 0]
    spoof = [s for s, l in score_dict.values() if l == 1]
    eer, _ = compute_eer(np.asarray(bona), np.asarray(spoof))
    return float(eer)


def fuse_scores_weighted(score_dicts: List[dict], weights: np.ndarray) -> dict[str, Tuple[float, int]]:
    fused: dict[str, Tuple[float, int]] = {}
    for utt_id in score_dicts[0]:
        ws = sum(w * sd[utt_id][0] for w, sd in zip(weights, score_dicts))
        fused[utt_id] = (float(ws), score_dicts[0][utt_id][1])
    return fused

###############################################################################
# Fusion techniques
###############################################################################
###############################################################################
# Differential‑Evolution optimisation
###############################################################################
def de_optimize_weights(
    score_dicts: List[dict],
    max_evals: int = 5000,
    seed: int | None = 42,
) -> np.ndarray:
    """Search the weight simplex for the EER‑optimal solution via SciPy
    Differential Evolution.  Requires scipy ≥ 1.7."""
    try:
        from scipy.optimize import differential_evolution
    except ModuleNotFoundError as exc:          # graceful fallback
        raise RuntimeError(
            "Fusion method 'de' needs SciPy (pip install scipy>=1.7)"
        ) from exc

    n_systems = len(score_dicts)
    bounds = [(0.0, 1.0)] * n_systems             # one bound per system

    # Objective: minimise EER; always renormalise to the probability simplex
    def _objective(w: np.ndarray) -> float:
        w = np.clip(w, 1e-9, None)
        w /= w.sum()
        fused = fuse_scores_weighted(score_dicts, w)
        return compute_eer_from_scores(fused)

    result = differential_evolution(
        _objective,
        bounds=bounds,
        strategy="best1bin",
        popsize=15,
        maxiter=max_evals // (15 * n_systems),   # heuristic cap on iterations
        seed=seed,
        tol=1e-5,
        polish=True,
        disp=False,
    )

    best_w = np.clip(result.x, 1e-9, None)
    best_w /= best_w.sum()                       # final renormalisation
    return best_w
def mean_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


# PSO, LOGREG, DE implementations unchanged (omitted for brevity in this comment)
# ... (keep the same bodies as previous revision) ...

###############################################################################
# CLI
###############################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spoof‑score fusion toolkit")
    p.add_argument("score_files", nargs="*", help="Score files (utt tag label score)")
    p.add_argument("--method", choices=["mean", "pso", "logreg", "de"], default="pso", help="Fusion technique")
    p.add_argument("--particles", type=int, default=30, help="[pso] particles")
    p.add_argument("--iters", type=int, default=150, help="[pso/logreg] iterations")
    p.add_argument("--max-evals", type=int, default=5000, help="[de] max function evaluations")
    p.add_argument("--out-file", default=DEFAULT_OUT_PATH, help="Path to save fused score file")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()

###############################################################################
# Main
###############################################################################

def main() -> None:
    args = parse_args()

    if not args.score_files:
        args.score_files = DEFAULT_SCORE_FILES.copy()
        print("[info] Using DEFAULT_SCORE_FILES:\n       " + "\n       ".join(args.score_files))

    score_dicts = [load_scores(p) for p in args.score_files]

    if args.method == "mean":
        w = mean_weights(len(score_dicts))
        fused = fuse_scores_weighted(score_dicts, w)
        print("Method : MEAN (equal weights)")
        print("Weights:", " ".join(f"{x:.4f}" for x in w))

    elif args.method == "pso":
        w = pso_optimize_weights(score_dicts, num_particles=args.particles, max_iter=args.iters, seed=args.seed)
        fused = fuse_scores_weighted(score_dicts, w)
        print("Method : PSO")
        print("Weights:", " ".join(f"{x:.4f}" for x in w))

    elif args.method == "logreg":
        w, b = logreg_optimize(score_dicts, max_iter=args.iters)
        fused = fuse_scores_logreg(score_dicts, w, b)
        print("Method : LOGREG")
        print("Weights:", " ".join(f"{x:.4f}" for x in w), " bias:", f"{b:.4f}")

    elif args.method == "de":
        w = de_optimize_weights(score_dicts, max_evals=args.max_evals, seed=args.seed)
        fused = fuse_scores_weighted(score_dicts, w)
        print("Method : DE (differential evolution)")
        print("Weights:", " ".join(f"{x:.4f}" for x in w))
    else:
        raise ValueError("Unknown method")

    eer = compute_eer_from_scores(fused)
   
    print(f"Fused EER: {eer:.4f}")

    # --- Save fused scores -------------------------------------------------
    write_fused_scores(args.score_files[0], fused, args.out_file)
    print(f"[info] Fused scores written to {args.out_file}")
    eer_cm, min_tDCF = compute_eer_and_tdcf( args.out_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    