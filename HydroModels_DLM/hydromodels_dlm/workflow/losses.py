import torch
from torch import nn


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------


def valid_train_mask(output, target, min_count=1):
    mask = ~torch.isnan(output) & ~torch.isnan(target)
    if mask.sum() < min_count:
        return None
    return mask


def zero_train_loss(output):
    return output.new_zeros((), requires_grad=True)


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------


class RMSELoss(nn.Module):
    def forward(self, output, target):
        mask = valid_train_mask(output, target, min_count=1)
        if mask is None:
            return zero_train_loss(output)
        sim = output[mask]
        obs = target[mask]
        return torch.sqrt(torch.mean((sim - obs) ** 2))


class NSELoss(nn.Module):
    def forward(self, output, target):
        mask = valid_train_mask(output, target, min_count=2)
        if mask is None:
            return zero_train_loss(output)
        sim = output[mask]
        obs = target[mask]
        denom = torch.sum((obs - obs.mean()) ** 2)
        if denom <= 0:
            return zero_train_loss(output)
        return torch.sum((sim - obs) ** 2) / denom


class HighRMSELoss(nn.Module):
    def forward(self, output, target):
        mask = valid_train_mask(output, target, min_count=1)
        if mask is None:
            return zero_train_loss(output)
        sim = output[mask]
        obs = target[mask]
        obs_peak = torch.max(obs)
        if obs_peak <= 0:
            return zero_train_loss(output)
        high_mask = obs >= obs_peak * 0.8
        if not high_mask.any():
            return zero_train_loss(output)
        return torch.sqrt(torch.mean((sim[high_mask] - obs[high_mask]) ** 2))


class PFELoss(nn.Module):
    def forward(self, output, target):
        mask = valid_train_mask(output, target, min_count=1)
        if mask is None:
            return zero_train_loss(output)
        sim = output[mask]
        obs = target[mask]
        obs_peak = torch.max(obs)
        if obs_peak <= 0:
            return zero_train_loss(output)
        return torch.abs(torch.max(sim) - obs_peak) / obs_peak


class PeakFocusedLoss(nn.Module):
    def __init__(
        self,
        peak_weight=2.0,
        high_weight=1.0,
        overall_weight=0.5,
    ):
        super().__init__()
        self.peak_weight = float(peak_weight)
        self.high_weight = float(high_weight)
        self.overall_weight = float(overall_weight)

    def forward(self, output, target):
        mask = valid_train_mask(output, target, min_count=1)
        if mask is None:
            return zero_train_loss(output)
        sim = output[mask]
        obs = target[mask]

        sq_err = (sim - obs) ** 2
        overall_rmse = torch.sqrt(torch.mean(sq_err))

        obs_peak = torch.max(obs)
        sim_peak = torch.max(sim)
        pfe = torch.abs(sim_peak - obs_peak) / torch.clamp(obs_peak, min=1e-12)
        pfe = torch.where(obs_peak > 0, pfe, torch.zeros_like(pfe))

        high_mask = (obs >= obs_peak * 0.8).to(sq_err.dtype)
        high_rmse = torch.sqrt((sq_err * high_mask).sum() / high_mask.sum().clamp(min=1.0))
        high_rmse = torch.where(obs_peak > 0, high_rmse, torch.zeros_like(high_rmse))

        return (
            self.overall_weight * overall_rmse
            + self.peak_weight * pfe
            + self.high_weight * high_rmse
        )


LOSS_FUNCTIONS = {
    'RMSE': RMSELoss,
    'NSE': NSELoss,
    'HIGHRMSE': HighRMSELoss,
    'PFE': PFELoss,
    'PEAKFOCUSED': PeakFocusedLoss,
}
