"""Command-line entry point.

Examples
--------
Train on Cora with default config:
    python main.py train --config configs/cora.yaml

Evaluate from a checkpoint:
    python main.py evaluate --config configs/cora.yaml \\
        --ckpt experiments/runs/cora_seed0_K2/best.pt

Run an ablation:
    python main.py ablate --config configs/yelpchi.yaml --ablation no_str

Visualise the reward landscape:
    python main.py visualize --config configs/yelpchi.yaml \\
        --ckpt experiments/runs/yelpchi_seed0_K2/best.pt \\
        --out figure_reward_landscape.pdf
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from evaluate import EvalConfig, evaluate
from models.irl_gad import IRLGADConfig
from train import TrainConfig, train
from utils.data_utils import DatasetSpec, load_dataset


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_train_cfg(yaml_cfg: dict[str, Any], overrides: dict[str, Any]) -> TrainConfig:
    ds_cfg = yaml_cfg.get("dataset", {})
    mdl_cfg = yaml_cfg.get("model", {})
    trn_cfg = yaml_cfg.get("training", {})
    # apply overrides keyed as "section.field=value"
    for k, v in overrides.items():
        sect, field = k.split(".", 1)
        target = {"dataset": ds_cfg, "model": mdl_cfg, "training": trn_cfg}[sect]
        target[field] = v

    return TrainConfig(
        dataset=DatasetSpec(**ds_cfg),
        model=IRLGADConfig(in_dim=1, **mdl_cfg),  # in_dim filled in by trainer
        **trn_cfg,
    )


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------
def _cmd_train(args):
    yaml_cfg = _load_yaml(args.config)
    overrides = _parse_kv(args.set)
    cfg = _build_train_cfg(yaml_cfg, overrides)
    out = train(cfg)
    print(json.dumps(out, indent=2, default=str))


def _cmd_evaluate(args):
    yaml_cfg = _load_yaml(args.config)
    cfg = EvalConfig(
        dataset=DatasetSpec(**yaml_cfg.get("dataset", {})),
        ckpt=args.ckpt,
        seed=yaml_cfg.get("training", {}).get("seed", 0),
        score_mode=args.score_mode,
    )
    out = evaluate(cfg)
    print(json.dumps(out, indent=2, default=str))


def _cmd_ablate(args):
    yaml_cfg = _load_yaml(args.config)
    overrides: dict[str, Any] = {}
    a = args.ablation
    if a == "no_str":
        overrides["model.use_str"] = False
    elif a == "no_sem":
        overrides["model.use_sem"] = False
    elif a == "no_tmp":
        overrides["model.use_tmp"] = False
    elif a == "single_reward":
        overrides["model.use_str"] = True
        overrides["model.use_sem"] = False
        overrides["model.use_tmp"] = False
    elif a == "recon_score":
        overrides["model.score_mode"] = "reconstruction"
    elif a == "contrast_score":
        overrides["model.score_mode"] = "contrastive"
    else:
        raise ValueError(f"Unknown ablation: {a}")

    cfg = _build_train_cfg(yaml_cfg, overrides)
    cfg.run_name = f"{cfg.dataset.name}_ablate_{a}_seed{cfg.seed}"
    out = train(cfg)
    print(json.dumps(out, indent=2, default=str))


def _cmd_visualize(args):
    from models.irl_gad import IRLGAD, IRLGADConfig
    from utils.visualization import reward_landscape_tsne

    yaml_cfg = _load_yaml(args.config)
    spec = DatasetSpec(**yaml_cfg.get("dataset", {}))
    data = load_dataset(spec)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    blob = torch.load(args.ckpt, map_location=device)
    model_cfg = IRLGADConfig(**blob["model_cfg"])
    model = IRLGAD(model_cfg).to(device)
    state = {k: v.to(device) for k, v in blob["state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        scored = model.score(data.x, data.edge_index)
        h = scored["h_final"]
        rfeat = scored["reward_feat"]

    reward_landscape_tsne(
        embeddings=h,
        reward_features=rfeat,
        y_anom=data.y_anom,
        out_path=args.out,
    )
    print(f"figure -> {args.out}")


# ---------------------------------------------------------------------------
# argparse plumbing
# ---------------------------------------------------------------------------
def _parse_kv(items: list[str] | None) -> dict[str, Any]:
    """Parse --set key=value pairs into a dict; types inferred."""
    out: dict[str, Any] = {}
    if not items:
        return out
    for it in items:
        k, _, v = it.partition("=")
        # try int -> float -> bool -> string
        for cast in (int, float):
            try:
                out[k] = cast(v); break
            except ValueError:
                continue
        else:
            if v.lower() in ("true", "false"):
                out[k] = (v.lower() == "true")
            else:
                out[k] = v
    return out


def main():
    p = argparse.ArgumentParser("IRL-GAD")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--set", nargs="*", default=[],
                   help="config overrides, e.g. --set training.lr=5e-4")
    t.set_defaults(fn=_cmd_train)

    e = sub.add_parser("evaluate")
    e.add_argument("--config", required=True)
    e.add_argument("--ckpt",   required=True)
    e.add_argument("--score-mode", default=None,
                   choices=[None, "irl", "reconstruction", "contrastive"])
    e.set_defaults(fn=_cmd_evaluate)

    a = sub.add_parser("ablate")
    a.add_argument("--config",   required=True)
    a.add_argument("--ablation", required=True,
                   choices=["no_str","no_sem","no_tmp",
                            "single_reward","recon_score","contrast_score"])
    a.set_defaults(fn=_cmd_ablate)

    v = sub.add_parser("visualize")
    v.add_argument("--config", required=True)
    v.add_argument("--ckpt",   required=True)
    v.add_argument("--out",    default="figure_reward_landscape.pdf")
    v.set_defaults(fn=_cmd_visualize)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
