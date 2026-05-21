"""
gradient_analysis.py
--------------------
Plug-in gradient diagnostics for RewardTrainer.
No matplotlib dependency — all output is printed to stdout.

Usage
-----
from gradient_analysis import GradientAnalyser

analyser = GradientAnalyser(model, criterion)
analyser.run_full_report(sample_inputs)   # one-shot diagnostic
# -- or use hooks during normal training --
analyser.register_hooks()
trainer.train()
analyser.print_summary()
analyser.print_norm_over_time()
analyser.remove_hooks()
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ascii_bar(value: float, vmin: float, vmax: float, width: int = 20) -> str:
    """Render a single ASCII bar scaled between vmin and vmax."""
    if vmax <= vmin:
        frac = 0.0
    else:
        frac = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def _sparkline(values: list[float], width: int = 40) -> str:
    """Compress a list of floats into a fixed-width ASCII sparkline."""
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    arr = np.array(values, dtype=float)
    # Work in log space so tiny/large values are both visible
    arr = np.log10(np.clip(arr, 1e-12, None))
    vmin, vmax = arr.min(), arr.max()
    # Downsample to `width` characters
    indices = np.linspace(0, len(arr) - 1, width).astype(int)
    sampled = arr[indices]
    if vmax > vmin:
        normalised = (sampled - vmin) / (vmax - vmin)
    else:
        normalised = np.zeros_like(sampled)
    chars = [blocks[int(round(v * (len(blocks) - 1)))] for v in normalised]
    return "".join(chars)


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class GradientAnalyser:
    """
    Attaches backward hooks to every named parameter and records gradient norms
    across batches. All output goes to stdout — no matplotlib required.
    """

    def __init__(self, model: nn.Module, criterion: nn.Module):
        self.model = model
        self.criterion = criterion
        self._hooks: list = []
        self.grad_norms: dict[str, list[float]] = defaultdict(list)
        self.grad_max:   dict[str, list[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_hooks(self) -> None:
        """Register backward hooks on every leaf parameter."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                def make_hook(n):
                    def hook(grad):
                        if grad is not None:
                            self.grad_norms[n].append(grad.norm().item())
                            self.grad_max[n].append(grad.abs().max().item())
                    return hook
                h = param.register_hook(make_hook(name))
                self._hooks.append(h)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def layer_norm_summary(self) -> dict[str, dict[str, float]]:
        summary = {}
        for name, norms in self.grad_norms.items():
            arr = np.array(norms)
            summary[name] = {
                "mean":     float(arr.mean()),
                "std":      float(arr.std()),
                "min":      float(arr.min()),
                "max":      float(arr.max()),
                "dead_pct": float((arr < 1e-8).mean() * 100),
            }
        return summary

    def print_summary(self) -> None:
        """
        Prints a table of per-parameter gradient stats plus an ASCII bar
        showing the relative magnitude of each layer.
        """
        summary = self.layer_norm_summary()
        if not summary:
            print("No gradient data recorded. Did you call register_hooks()?")
            return

        means = [s["mean"] for s in summary.values()]
        log_means = [math.log10(max(m, 1e-12)) for m in means]
        lmin, lmax = min(log_means), max(log_means)

        W = 72
        print(f"\n{'═' * W}")
        print("  Layer-wise Gradient Norm Summary")
        print(f"{'═' * W}")
        hdr = f"{'Parameter':<38} {'Mean':>9} {'Std':>9} {'Dead%':>6}  {'Magnitude (log scale)'}"
        print(hdr)
        print(f"{'─' * W}")

        for (name, s), lm in zip(summary.items(), log_means):
            bar   = _ascii_bar(lm, lmin, lmax, width=18)
            flag  = " ⚠ VANISHING" if s["dead_pct"] > 20 else (
                    " 🔥 EXPLODING" if s["mean"] > 1e2 else "")
            short = name if len(name) <= 38 else "…" + name[-37:]
            print(
                f"{short:<38} {s['mean']:>9.2e} {s['std']:>9.2e} "
                f"{s['dead_pct']:>5.1f}%  [{bar}]{flag}"
            )

        print(f"{'─' * W}")
        print(f"  Bar scale: left = {10**lmin:.1e}  →  right = {10**lmax:.1e}")
        print(f"{'═' * W}\n")

    def print_norm_over_time(
        self,
        param_names: Optional[list[str]] = None,
        width: int = 50,
    ) -> None:
        """
        Prints a sparkline of gradient norm over batches for each parameter.
        Pass param_names to show a subset; defaults to all recorded params.
        """
        names = param_names or list(self.grad_norms.keys())
        W = 72
        print(f"\n{'═' * W}")
        print("  Gradient Norm Over Training Batches  (log scale sparkline)")
        print(f"{'═' * W}")
        print(f"  Each character = {width} samples compressed into 1 column")
        print(f"  ▁ = low  →  █ = high  (relative to that layer's own range)")
        print(f"{'─' * W}")

        for name in names:
            if name not in self.grad_norms:
                print(f"  WARNING: '{name}' not in recorded hooks — skipped.")
                continue
            norms = self.grad_norms[name]
            spark = _sparkline(norms, width=width)
            arr   = np.array(norms)
            short = name if len(name) <= 35 else "…" + name[-34:]
            print(f"  {short:<35} [{spark}]  ({arr.min():.1e}–{arr.max():.1e})")

        print(f"{'═' * W}\n")

    # ------------------------------------------------------------------
    # Criterion gradient probe
    # ------------------------------------------------------------------

    def probe_criterion_gradients(
        self,
        sample_inputs: torch.Tensor,
        tau_values: Optional[list[float]] = None,
    ) -> None:
        """
        Passes a dummy zero-logit grid through the criterion and prints the
        gradient w.r.t. the grid at different tau values.

        Isolates whether soft_run_lengths is differentiable at the values your
        model likely produces early in training — before any model weights matter.
        """
        tau_values = tau_values or [0.1, 0.3, 0.5, 1.0]
        device = next(self.model.parameters()).device
        sample_inputs = sample_inputs.to(device)

        N    = sample_inputs.shape[0]
        side = math.isqrt(sample_inputs.shape[-1] // 2)
        n_out = side * side

        W = 72
        print(f"\n{'═' * W}")
        print("  Criterion Gradient Probe  (dummy grid, all logits = 0.0)")
        print(f"{'═' * W}")
        print(f"  {'tau':>5}  {'‖∇grid‖':>12}  {'max|∇|':>12}  {'zero%':>7}  Status")
        print(f"{'─' * W}")

        original_tau = self.criterion.tau
        for tau in tau_values:
            self.criterion.tau = tau
            grid_logits = torch.zeros(N, n_out, requires_grad=True, device=device)
            loss, _, _ = self.criterion(grid_logits, sample_inputs)
            loss.backward()

            g = grid_logits.grad
            if g is None:
                print(f"  {tau:>5.2f}  {'NO GRADIENT':>12}")
                continue

            l2   = g.norm().item()
            gmax = g.abs().max().item()
            zero = (g.abs() < 1e-10).float().mean().item() * 100

            if l2 < 1e-6:
                status = "⚠  VANISHING — loss won't train model at this tau"
            elif l2 > 1e3:
                status = "🔥 EXPLODING — reduce tau or add grad clipping"
            else:
                status = "✓  OK"

            print(f"  {tau:>5.2f}  {l2:>12.4e}  {gmax:>12.4e}  {zero:>6.1f}%  {status}")

        self.criterion.tau = original_tau
        print(f"{'─' * W}")
        print(f"  Grid shape: {N} × {side}×{side}")
        print(f"{'═' * W}\n")

    # ------------------------------------------------------------------
    # One-shot full report
    # ------------------------------------------------------------------

    def run_full_report(
        self,
        sample_inputs: torch.Tensor,
        tau_values: Optional[list[float]] = None,
    ) -> None:
        """
        Runs every diagnostic in one call:
          1. Criterion gradient probe (tau sweep)
          2. Single forward+backward through the real model
          3. Per-layer gradient norm summary
          4. Sparkline over that single backward pass (1 step, so trivial here;
             more useful when called after trainer.train() with hooks active)
        """
        device = next(self.model.parameters()).device
        sample_inputs = sample_inputs.to(device)

        # 1. Criterion probe — is the loss itself differentiable?
        self.probe_criterion_gradients(sample_inputs, tau_values)

        # 2. One real forward/backward with hooks
        self.register_hooks()
        self.model.train()
        self.model.zero_grad()

        outputs = self.model(sample_inputs)
        loss, _, _ = self.criterion(outputs, sample_inputs)
        loss.backward()

        # 3 & 4. Print results
        self.print_summary()

        self.remove_hooks()