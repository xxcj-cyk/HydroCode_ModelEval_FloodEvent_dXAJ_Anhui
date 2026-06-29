"""Differentiable GR4J (production + routing) with batched GPU-friendly ops."""

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Production store
# ---------------------------------------------------------------------------


def precip_store_in(s, precip_net, x1):
    return (
        x1 * (1.0 - (s / x1) ** 2) * torch.tanh(precip_net / x1)
    ) / (1.0 + (s / x1) * torch.tanh(precip_net / x1))


def evap_store_out(s, evap_net, x1):
    return (
        s * (2.0 - s / x1) * torch.tanh(evap_net / x1)
    ) / (1.0 + (1.0 - s / x1) * torch.tanh(evap_net / x1))


def percolation(store, x1):
    return store * (1.0 - (1.0 + (4.0 / 9.0 * store / x1) ** 4) ** -0.25)


def clamp_store(s, x1):
    return torch.minimum(torch.maximum(s, torch.zeros_like(s)), x1)


def production_step(p, e, x1, s):
    diff = p - e
    pn = torch.clamp(diff, min=0.0)
    en = torch.clamp(-diff, min=0.0)
    s = clamp_store(s, x1)
    ps = precip_store_in(s, pn, x1)
    es = evap_store_out(s, en, x1)
    s = clamp_store(s - es + ps, x1)
    perc = percolation(s, x1)
    s = s - perc
    return perc + (pn - ps), es, s


# ---------------------------------------------------------------------------
# Unit hydrograph (batched, differentiable)
# ---------------------------------------------------------------------------


def s_curve_90_batched(t, x4):
    return torch.where(
        t <= 0,
        torch.zeros_like(t),
        torch.where(t < x4, (t / x4) ** 2.5, torch.ones_like(t)),
    )


def s_curve_10_batched(t, x4):
    ratio = torch.clamp(2.0 - t / x4, min=0.0)
    mid = 1.0 - 0.5 * ratio ** 2.5
    return torch.where(
        t <= 0,
        torch.zeros_like(t),
        torch.where(
            t < x4,
            0.5 * (t / x4) ** 2.5,
            torch.where(t < 2 * x4, mid, torch.ones_like(t)),
        ),
    )


def unit_hydrograph_weights_batched(x4, *, quickflow_90):
    """Return padded UH weights with shape [B, M_max]."""
    x4 = x4.reshape(-1)
    if x4.numel() == 0:
        return x4.new_zeros(0, 0)
    if quickflow_90:
        max_n = int(torch.ceil(x4.max()).item())
    else:
        max_n = int(torch.ceil(2.0 * x4.max()).item())
    if max_n <= 0:
        return x4.new_zeros(x4.shape[0], 0)

    t = torch.arange(1, max_n + 1, device=x4.device, dtype=x4.dtype).view(1, -1)
    x4_col = x4.view(-1, 1)
    if quickflow_90:
        w = s_curve_90_batched(t, x4_col) - s_curve_90_batched(t - 1, x4_col)
    else:
        w = s_curve_10_batched(t, x4_col) - s_curve_10_batched(t - 1, x4_col)
    return w


def convolve_truncated_batched(x, weights):
    """Causal FIR along time for all basins; x [T,B], weights [B,M]."""
    if weights.shape[-1] == 0:
        return x.new_zeros(x.shape)
    weights = weights.to(device=x.device, dtype=x.dtype)
    t_steps, n_basin = x.shape
    kernel = weights.flip(-1).unsqueeze(1)
    inp = x.transpose(0, 1).unsqueeze(0)
    padded = F.pad(inp, (kernel.shape[-1] - 1, 0))
    out = F.conv1d(padded, kernel, groups=n_basin)
    return out.squeeze(0).transpose(0, 1)[:t_steps]


# ---------------------------------------------------------------------------
# Routing store
# ---------------------------------------------------------------------------


def route_store_step(q9, q1, x2, x3, r):
    r = torch.minimum(torch.maximum(r, torch.zeros_like(r)), x3)
    exchange = x2 * (r / x3) ** 3.5
    r = torch.clamp(r + q9 + exchange, min=0.0)
    qr = r * (1.0 - (1.0 + (r / x3) ** 4) ** -0.25)
    r = r - qr
    qd = torch.clamp(q1 + exchange, min=0.0)
    return qr + qd, r


# ---------------------------------------------------------------------------
# Simulation (time scans are torch.compile-friendly)
# ---------------------------------------------------------------------------


def scan_production(p, e, x1, s_init):
    n_time = p.shape[0]
    prs = p.new_zeros(n_time, p.shape[1])
    s = s_init
    for t in range(n_time):
        prs[t], _, s = production_step(p[t], e[t], x1, s)
    return prs, s


def scan_routing(q9, q1, x2, x3, r_init):
    n_time = q9.shape[0]
    qsim = q9.new_zeros(n_time, q9.shape[1])
    r = r_init
    for t in range(n_time):
        qsim[t], r = route_store_step(q9[t], q1[t], x2, x3, r)
    return qsim, r


def effective_grad_steps(grad_steps, n_time):
    if grad_steps is None or int(grad_steps) <= 0:
        return n_time
    steps = int(grad_steps)
    if steps >= n_time:
        return n_time
    return steps


def prefix_production_no_grad(p, e, x1, s_init):
    with torch.no_grad():
        return scan_production(p, e, x1, s_init)


def prefix_production_numba(p, e, x1, s_init, device, dtype):
    from hydromodels_dlm.model.physics_guided import numba_available, prefix_gr4j_production

    if not numba_available():
        return prefix_production_no_grad(p, e, x1, s_init)
    prs_np, s_np = prefix_gr4j_production(p, e, x1, s_init=s_init)
    prs = torch.as_tensor(prs_np, device=device, dtype=dtype)
    s = torch.as_tensor(s_np, device=device, dtype=dtype)
    return prs, s


def prefix_routing_no_grad(q9, q1, x2, x3, r_init):
    with torch.no_grad():
        return scan_routing(q9, q1, x2, x3, r_init)


def prefix_routing_numba(prs, x2, x3, x4, r_init, device, dtype):
    from hydromodels_dlm.model.physics_guided import numba_available, prefix_gr4j_routing

    if not numba_available():
        w90 = unit_hydrograph_weights_batched(x4, quickflow_90=True)
        w10 = unit_hydrograph_weights_batched(x4, quickflow_90=False)
        q9 = convolve_truncated_batched(prs, w90)
        q1 = convolve_truncated_batched(prs, w10)
        return prefix_routing_no_grad(q9, q1, x2, x3, r_init)

    qsim_np, r_np = prefix_gr4j_routing(prs, x2, x3, x4, r_init=r_init)
    qsim = torch.as_tensor(qsim_np, device=device, dtype=dtype)
    r = torch.as_tensor(r_np, device=device, dtype=dtype)
    return qsim, r


def simulate_dGR4J(
    p_and_e,
    x1,
    x2,
    x3,
    x4,
    *,
    grad_steps=None,
):
    dtype = p_and_e.dtype
    device = p_and_e.device
    x1 = x1.to(dtype=dtype)
    x2 = x2.to(dtype=dtype)
    x3 = x3.to(dtype=dtype)
    x4 = x4.to(dtype=dtype)
    p = p_and_e[:, :, 0]
    e = p_and_e[:, :, 1]
    n_time = p.shape[0]
    steps = effective_grad_steps(grad_steps, n_time)
    if steps >= n_time:
        prs, _ = scan_production(p, e, x1, 0.3 * x1)
        w90 = unit_hydrograph_weights_batched(x4, quickflow_90=True)
        w10 = unit_hydrograph_weights_batched(x4, quickflow_90=False)
        q9 = convolve_truncated_batched(prs, w90)
        q1 = convolve_truncated_batched(prs, w10)
        qsim, _ = scan_routing(q9, q1, x2, x3, 0.5 * x3)
        return qsim

    t0 = n_time - steps
    s0 = 0.3 * x1
    r0 = 0.5 * x3
    prs_pre, s_mid = prefix_production_numba(
        p[:t0], e[:t0], x1, s0, device, dtype
    )

    prs_post, _ = scan_production(p[t0:], e[t0:], x1, s_mid.detach())
    prs = torch.cat([prs_pre.detach(), prs_post], dim=0)

    w90 = unit_hydrograph_weights_batched(x4, quickflow_90=True)
    w10 = unit_hydrograph_weights_batched(x4, quickflow_90=False)
    q9 = convolve_truncated_batched(prs, w90)
    q1 = convolve_truncated_batched(prs, w10)

    qsim_pre, r_mid = prefix_routing_numba(prs[:t0], x2, x3, x4, r0, device, dtype)

    qsim_post, _ = scan_routing(q9[t0:], q1[t0:], x2, x3, r_mid.detach())
    return torch.cat([qsim_pre.detach(), qsim_post], dim=0)


# ---------------------------------------------------------------------------
# PBM core
# ---------------------------------------------------------------------------


class GR4JCore(nn.Module):
    driver_dim = 2
    pbm_name = 'dGR4J'

    def __init__(
        self,
        warmup_length=365,
        *,
        pbm_grad_steps=None,
    ):
        super().__init__()
        self.warmup_length = int(warmup_length)
        self.pbm_grad_steps = pbm_grad_steps

    def forward(self, drivers, params):
        if drivers.shape[-1] != self.driver_dim:
            raise ValueError(
                f'GR4JCore expects {self.driver_dim} drivers, got {drivers.shape[-1]}'
            )
        qsim = simulate_dGR4J(
            drivers,
            params[:, 0],
            params[:, 1],
            params[:, 2],
            params[:, 3],
            grad_steps=self.pbm_grad_steps,
        )
        return qsim.unsqueeze(-1)
