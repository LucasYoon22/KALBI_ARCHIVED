#!/usr/bin/env python3

import math

import torch

from .train import TrainerError
from .train_selector_only_v23_frozen_kalbi_v6_fhatfzero_8gpu import (
    selector_heldout_fhat_fzero_loss,
)


def main() -> None:
    target = torch.tensor([[1.0, -1.0]])
    prediction = torch.zeros_like(target, requires_grad=True)
    loss, f_hat, f_zero, ratio = selector_heldout_fhat_fzero_loss(
        prediction,
        target,
    )
    torch.testing.assert_close(f_hat, torch.tensor([0.5]))
    torch.testing.assert_close(f_zero, torch.tensor([0.5]))
    torch.testing.assert_close(ratio, torch.tensor([1.0]))
    torch.testing.assert_close(loss, torch.tensor(math.log(2.0)))
    if torch.isclose(loss, torch.tensor(math.log(3.0))):
        raise AssertionError("objective incorrectly implements log1p(1 + ratio)")
    loss.backward()
    if prediction.grad is None or not bool(prediction.grad.abs().sum() > 0):
        raise AssertionError("normalized held-out objective gradient is missing")

    scaled_loss, _, _, scaled_ratio = selector_heldout_fhat_fzero_loss(
        torch.zeros_like(target),
        7.0 * target,
    )
    torch.testing.assert_close(scaled_ratio, ratio)
    torch.testing.assert_close(scaled_loss, loss.detach())

    exact_loss, exact_f_hat, _, _ = selector_heldout_fhat_fzero_loss(
        target,
        target,
    )
    torch.testing.assert_close(exact_f_hat, torch.zeros_like(exact_f_hat))
    torch.testing.assert_close(exact_loss, torch.zeros_like(exact_loss))

    try:
        selector_heldout_fhat_fzero_loss(
            torch.zeros((1, 4)),
            torch.zeros((1, 4)),
        )
    except TrainerError:
        pass
    else:
        raise AssertionError("degenerate F_zero was not rejected")

    print(
        {
            "status": "PASS",
            "formula": "mean(log1p(F_hat/F_zero))",
            "unit_ratio_loss": float(loss.detach()),
            "scale_invariant": True,
            "gradient_nonzero": True,
            "degenerate_f_zero_rejected": True,
        }
    )


if __name__ == "__main__":
    main()
