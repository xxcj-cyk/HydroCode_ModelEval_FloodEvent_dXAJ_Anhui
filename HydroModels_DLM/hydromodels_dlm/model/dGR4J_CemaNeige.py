import torch
from torch import nn

from hydromodels_dlm.model.dGR4J import effective_grad_steps, simulate_dGR4J


def frac_solid_usace(temp):
    tmin = -1.0
    tmax = 3.0
    return torch.where(
        temp <= tmin,
        torch.ones_like(temp),
        torch.where(
            temp >= tmax,
            torch.zeros_like(temp),
            1.0 - (temp - tmin) / (tmax - tmin),
        ),
    )


def cemaneige_step(p, temp, cn1, cn2, g, etg, gthreshold):
    frac = frac_solid_usace(temp)
    pliq = (1.0 - frac) * p
    g = g + frac * p
    etg = cn1 * etg + (1.0 - cn1) * temp
    etg = torch.clamp(etg, max=0.0)
    pot_melt = torch.where(
        (etg == 0.0) & (temp > 0.0),
        torch.clamp(cn2 * temp, max=g),
        torch.zeros_like(g),
    )
    gratio = torch.where(g < gthreshold, g / gthreshold, torch.ones_like(g))
    melt = (0.9 * gratio + 0.1) * pot_melt
    g = g - melt
    return pliq + melt, g, etg


def scan_cemaneige(p, temp, cn1, cn2, gthreshold, *, g_init=None, etg_init=None):
    n_time = p.shape[0]
    pliq = p.new_zeros(p.shape[0], p.shape[1])
    g = p.new_zeros(p.shape[1]) if g_init is None else g_init
    etg = p.new_zeros(p.shape[1]) if etg_init is None else etg_init
    for t in range(n_time):
        pliq[t], g, etg = cemaneige_step(
            p[t], temp[t], cn1, cn2, g, etg, gthreshold
        )
    return pliq, g, etg


def prefix_cemaneige_no_grad(p, temp, cn1, cn2, gthreshold):
    with torch.no_grad():
        return scan_cemaneige(p, temp, cn1, cn2, gthreshold)


def prefix_cemaneige_numba(p, temp, cn1, cn2, gthreshold, device, dtype):
    from hydromodels_dlm.model.physics_guided import numba_available, prefix_cemaneige_states

    if not numba_available():
        return prefix_cemaneige_no_grad(p, temp, cn1, cn2, gthreshold)
    pliq_np, g_np, etg_np = prefix_cemaneige_states(p, temp, cn1, cn2, gthreshold)
    pliq = torch.as_tensor(pliq_np, device=device, dtype=dtype)
    g = torch.as_tensor(g_np, device=device, dtype=dtype)
    etg = torch.as_tensor(etg_np, device=device, dtype=dtype)
    return pliq, g, etg


def cemaneige_with_grad_steps(
    p,
    temp,
    cn1,
    cn2,
    gthreshold,
    *,
    grad_steps=None,
):
    n_time = p.shape[0]
    steps = effective_grad_steps(grad_steps, n_time)
    if steps >= n_time:
        pliq, _, _ = scan_cemaneige(p, temp, cn1, cn2, gthreshold)
        return pliq

    t0 = n_time - steps
    device, dtype = p.device, p.dtype
    pliq_pre, g_mid, etg_mid = prefix_cemaneige_numba(
        p[:t0], temp[:t0], cn1, cn2, gthreshold, device, dtype
    )
    pliq_post, _, _ = scan_cemaneige(
        p[t0:],
        temp[t0:],
        cn1,
        cn2,
        gthreshold,
        g_init=g_mid.detach(),
        etg_init=etg_mid.detach(),
    )
    return torch.cat([pliq_pre.detach(), pliq_post], dim=0)


def gthreshold_from_warmup(p, temp, warmup_length):
    n_time = p.shape[0]
    n_clim = int(warmup_length)
    if n_clim <= 0 or n_clim > n_time:
        n_clim = n_time
    if n_clim <= 0:
        return torch.ones(p.shape[1], device=p.device, dtype=p.dtype)
    frac = frac_solid_usace(temp[:n_clim])
    mean_solid = (frac * p[:n_clim]).mean(dim=0) * 365.25
    val = 0.9 * mean_solid
    return torch.where(val <= 0, torch.ones_like(val), val)


def gthreshold_climatology(p, temp, warmup_length):
    """Fixed gthreshold from the first ``warmup_length`` days (PBM-style)."""
    p_col = p.reshape(-1, 1) if p.ndim == 1 else p
    temp_col = temp.reshape(-1, 1) if temp.ndim == 1 else temp
    return gthreshold_from_warmup(p_col, temp_col, warmup_length)


class GR4JCemaNeigeCore(nn.Module):
    """CemaNeige snow + GR4J (P, PET, TEMP drivers)."""

    driver_dim = 3
    pbm_name = 'dGR4J-CemaNeige'

    def __init__(
        self,
        warmup_length=365,
        *,
        pbm_grad_steps=None,
    ):
        super().__init__()
        self.warmup_length = int(warmup_length)
        self.pbm_grad_steps = pbm_grad_steps
        self.register_buffer('gthreshold_const', torch.empty(0), persistent=True)

    def set_gthreshold_climatology(self, p, temp, warmup_length=None):
        """Use train-period climatology (fixed), matching HydroModels_PBM."""
        wl = int(warmup_length or self.warmup_length)
        p_arr = torch.as_tensor(p, dtype=torch.float32)
        temp_arr = torch.as_tensor(temp, dtype=torch.float32)
        if p_arr.ndim == 1:
            p_arr = p_arr.unsqueeze(-1)
            temp_arr = temp_arr.unsqueeze(-1)
        gt = gthreshold_climatology(p_arr[:wl], temp_arr[:wl], wl).reshape(-1)
        if self.gthreshold_const.numel() == 0:
            self.register_buffer('gthreshold_const', gt.clone())
        else:
            self.gthreshold_const.copy_(gt)

    def forward(self, drivers, params):
        if drivers.shape[-1] != self.driver_dim:
            raise ValueError(
                f'GR4JCemaNeigeCore expects {self.driver_dim} drivers, got {drivers.shape[-1]}'
            )
        p = drivers[:, :, 0]
        pet = drivers[:, :, 1]
        temp = drivers[:, :, 2]
        if self.gthreshold_const.numel() > 0:
            gthreshold = self.gthreshold_const.to(device=p.device, dtype=p.dtype).reshape(-1)
            if gthreshold.shape[0] == 1 and params.shape[0] > 1:
                gthreshold = gthreshold.expand(params.shape[0])
        else:
            gthreshold = gthreshold_from_warmup(p, temp, self.warmup_length)
        pliq = cemaneige_with_grad_steps(
            p,
            temp,
            params[:, 4],
            params[:, 5],
            gthreshold,
            grad_steps=self.pbm_grad_steps,
        )
        p_and_e = drivers.new_empty(drivers.shape[0], drivers.shape[1], 2)
        p_and_e[:, :, 0] = pliq
        p_and_e[:, :, 1] = pet
        qsim = simulate_dGR4J(
            p_and_e,
            params[:, 0],
            params[:, 1],
            params[:, 2],
            params[:, 3],
            grad_steps=self.pbm_grad_steps,
        )
        return qsim.unsqueeze(-1)
