"""Run every experiment in the README and write CSVs to results/.

    python scripts/run_experiments.py              # everything (about 1.5 h on 2 CPU cores)
    python scripts/run_experiments.py --quick      # a 10-minute smoke version
    python scripts/run_experiments.py --only main  # just the main comparison and ablations

Finished runs are skipped, so you can stop and resume.
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEARNED = ["airfed", "no_gat", "no_reputation", "no_fed", "no_quant", "independent"]
BASELINES = ["random", "static_local", "static_balanced"]
SCALE = [(4, 40), (8, 40), (6, 20), (6, 60)]          # (UAVs, devices)
SCALE_METHODS = ["airfed", "static_balanced", "random"]


def jobs(quick, only):
    episodes = 20 if quick else 100
    seeds = [0] if quick else [0, 1, 2]
    out = []
    if only in (None, "main"):
        for seed in seeds:
            for m in LEARNED + BASELINES:
                out.append(dict(method=m, k=6, m=40, seed=seed, episodes=episodes))
    if only in (None, "scale"):
        for seed in seeds[:2]:
            for k, m in SCALE:
                for method in SCALE_METHODS:
                    out.append(dict(method=method, k=k, m=m, seed=seed, episodes=episodes))
    return out


def run(job):
    import torch
    from airfed.config import Config
    from airfed.train import EVAL_TASKS, evaluate, run_baseline, save, train

    torch.set_num_threads(1)
    tag = f"{job['method']}_k{job['k']}_m{job['m']}_s{job['seed']}.csv"
    train_csv, eval_csv = ROOT / "results/runs" / tag, ROOT / "results/eval" / tag
    if train_csv.exists() and eval_csv.exists():
        return f"skip {tag}"

    cfg = Config().with_env(n_uavs=job["k"], n_devices=job["m"])
    if job["method"] in BASELINES:
        rows = run_baseline(job["method"], job["episodes"], job["seed"], cfg)
        final = run_baseline(job["method"], 5, job["seed"], cfg, task_seed=job["seed"] + EVAL_TASKS)
    else:
        rows, agents = train(job["method"], job["episodes"], job["seed"], cfg, log=lambda s: None)
        final = evaluate(agents, job["method"], job["seed"], cfg)
    for r in rows + final:
        r.update(uavs=job["k"], devices=job["m"])
    save(rows, train_csv)
    save(final, eval_csv)
    return f"done {tag}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["main", "scale"])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    todo = jobs(args.quick, args.only)
    # long learned runs first so the pool stays busy
    todo.sort(key=lambda j: (j["method"] in BASELINES, -j["k"]))
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(run, todo):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
