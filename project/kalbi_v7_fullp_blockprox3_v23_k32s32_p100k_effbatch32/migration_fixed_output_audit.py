from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from train_fpk import _load_yaml

from .model import build_model
from .preflight_kalbi_v6_v20 import build_penalty


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not torch.is_tensor(tensor):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(314159)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model = build_model(_load_yaml(str(args.config)), "cpu")
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict load failed: {incompatible}")
    model.eval()

    rows = 48
    features = 64
    x = torch.randn(1, rows, features, dtype=torch.float32)
    beta_true = torch.zeros(1, features, dtype=torch.float32)
    beta_true[0, :8] = torch.linspace(1.0, -0.5, 8)
    y = torch.einsum("bnp,bp->bn", x, beta_true)
    y = y + 0.125 * torch.randn_like(y)

    with torch.no_grad():
        encoded = model.encode_dataset(model.precompute(x, y))
        lambda_max = encoded["lambda_max"].float()
        lambdas = lambda_max[:, None] * torch.tensor(
            [[0.025, 0.2]], dtype=torch.float32
        )
        penalty = build_penalty(torch.zeros(2, 96))
        result = model.decode_queries(
            encoded,
            lambdas=lambdas,
            lambda_max=lambda_max[:, None],
            penalty_shape_features=penalty["shape_features"].view(
                1, 2, 196
            ),
            penalty_d0=penalty["d0"].view(1, 2),
            return_aux=True,
        )
        beta = result["beta_q"].float().cpu().contiguous()

    payload = {
        "completed_steps": int(checkpoint["completed_steps"]),
        "model_state_sha256": state_dict_sha256(
            checkpoint["model_state_dict"]
        ),
        "beta_optimizer_state_count": len(
            checkpoint["beta_optimizer_state_dict"]["state"]
        ),
        "beta_scheduler_last_epoch": int(
            checkpoint["beta_scheduler_state_dict"]["last_epoch"]
        ),
        "beta_shape": list(beta.shape),
        "beta": beta.reshape(-1).tolist(),
    }
    print("MIGRATION_AUDIT_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
