import math
import torch
import torch.nn as nn


class AbstainLoss(nn.Module):
    """
    Cross-entropy loss with an abstain option.

    Args:
        abstain_penalty: Float in [0, 1]. Cost of fully abstaining relative to
            the loss for a uniform random guess. 0 = free abstention; 1 = abstain
            only when worse than random.
        entropy_weight: Weight for the entropy bonus that prevents collapse.
        eps: Numerical stability constant.
    """

    def __init__(
        self,
        abstain_penalty: float = 0.5,
        entropy_weight: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        if not (0.0 <= abstain_penalty <= 1.0):
            raise ValueError(f"abstain_penalty must be in [0, 1], got {abstain_penalty}")
        self.abstain_penalty = abstain_penalty
        self.entropy_weight = entropy_weight
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds:   [batch, num_classes + 1, cells]  (last channel = abstain)
        # targets: [batch, cells] with values in [0, num_classes - 1]

        num_classes = preds.size(1) - 1
        if num_classes < 2:
            raise ValueError("Need at least 2 non-abstain classes")

        # Scale so that 1.0 ≈ cost of a uniform random guess (-log(1/C))
        l_abs = self.abstain_penalty * math.log(num_classes)

        preds = torch.softmax(preds, dim=1)

        p_abs = preds[:, -1, :]                           # [batch, cells]
        p_non_abs = preds[:, :-1, :]                      # [batch, num_classes, cells]

        # Guard against targets accidentally indexing the abstain channel
        if targets.max() >= num_classes:
            raise ValueError("targets must be < num_classes")

        idx = targets.unsqueeze(1)                        # [batch, 1, cells]
        p_y = preds.gather(dim=1, index=idx).squeeze(1)    # [batch, cells]

        committed = (1.0 - p_abs).clamp(min=self.eps)
        p_y_renorm = (p_y / committed).clamp(min=self.eps, max=1.0)

        # Cross-entropy over the non-abstain distribution
        ce_loss = -torch.log(p_y_renorm)

        # Penalise abstention proportionally to how much probability it consumed
        abstain_cost = p_abs * l_abs

        # Entropy bonus over committed classes — discourages collapse
        entropy_bonus = -(p_non_abs * torch.log(p_non_abs.clamp(min=self.eps))).sum(dim=1)

        loss = ce_loss + abstain_cost - self.entropy_weight * entropy_bonus

        return loss.mean()
