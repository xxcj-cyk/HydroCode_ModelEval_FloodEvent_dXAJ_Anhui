import torch
from torch import nn

from hydromodels_dlm.model.dGR4J_CemaNeige import (
    cemaneige_with_grad_steps,
    gthreshold_climatology,
    gthreshold_from_warmup,
)
from hydromodels_dlm.model.dXAJ_mz import simulate_dXAJ_mz


class XAJMzCemaNeigeCore(nn.Module):
    """CemaNeige snow + XAJ-mz (P, PET, TEMP drivers)."""

    driver_dim = 3
    pbm_name = 'dXAJ-mz-CemaNeige'

    def __init__(
        self,
        warmup_length=365,
        *,
        kernel_size=15,
        pbm_grad_steps=None,
    ):
        super().__init__()
        self.warmup_length = int(warmup_length)
        self.kernel_size = int(kernel_size)
        self.pbm_grad_steps = pbm_grad_steps
        self.register_buffer('gthreshold_const', torch.empty(0), persistent=True)

    def set_gthreshold_climatology(self, p, temp, warmup_length=None):
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
                f'XAJMzCemaNeigeCore expects {self.driver_dim} drivers, '
                f'got {drivers.shape[-1]}'
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
            params[:, 15],
            params[:, 16],
            gthreshold,
            grad_steps=self.pbm_grad_steps,
        )
        p_and_e = drivers.new_empty(drivers.shape[0], drivers.shape[1], 2)
        p_and_e[:, :, 0] = pliq
        p_and_e[:, :, 1] = pet
        qsim = simulate_dXAJ_mz(
            p_and_e,
            params[:, :15],
            warmup_length=self.warmup_length,
            kernel_size=self.kernel_size,
            grad_steps=self.pbm_grad_steps,
        )
        return qsim.unsqueeze(-1)
