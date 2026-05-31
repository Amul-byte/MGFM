"""Flow matching utilities for latent skeleton generation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_model.util import DEFAULT_ODE_STEPS, assert_shape


def _broadcast_time(t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    return t.float().reshape(t.shape[0], *([1] * (len(x_shape) - 1)))


class FlowMatchingProcess(nn.Module):
    """OT straight-line flow matching process for normalized latent tensors."""

    diffusion_schema = "flow_matching_ot_v1"

    def __init__(self, ode_steps: int = DEFAULT_ODE_STEPS) -> None:
        super().__init__()
        self.ode_steps = int(ode_steps)

    def q_sample(
        self,
        x1: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample the OT interpolant x_t = t*x_1 + (1-t)*x_0."""
        assert_shape(x1, [None, None, None, None], "FlowMatchingProcess.q_sample.x1")
        assert_shape(t, [x1.shape[0]], "FlowMatchingProcess.q_sample.t")
        if noise is None:
            noise = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
        assert noise.shape == x1.shape, "noise and x1 shape mismatch"
        t_view = _broadcast_time(t, x1.shape).to(device=x1.device, dtype=x1.dtype)
        x_t = t_view * x1 + (1.0 - t_view) * noise
        assert x_t.shape == x1.shape, "q_sample output shape mismatch"
        return x_t

    def flow_matching_loss(
        self,
        denoiser: nn.Module,
        x1: torch.Tensor,
        t: torch.Tensor,
        h_tokens: torch.Tensor | None = None,
        h_global: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute velocity-prediction loss for OT flow matching."""
        assert_shape(x1, [None, None, None, None], "FlowMatchingProcess.flow_matching_loss.x1")
        noise = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
        x_t = self.q_sample(x1=x1, t=t, noise=noise)
        target_v = x1 - noise
        pred_v = denoiser(x_t, t, h_tokens=h_tokens, h_global=h_global)
        assert pred_v.shape == target_v.shape, "predicted velocity shape mismatch"
        loss = F.mse_loss(pred_v, target_v)
        return loss, pred_v, x_t, noise

    def predict_clean_from_velocity(self, x_t: torch.Tensor, t: torch.Tensor, pred_v: torch.Tensor) -> torch.Tensor:
        """Extrapolate from x_t to the clean endpoint x_1 using a predicted velocity."""
        assert_shape(x_t, [None, None, None, None], "FlowMatchingProcess.predict_clean_from_velocity.x_t")
        assert_shape(t, [x_t.shape[0]], "FlowMatchingProcess.predict_clean_from_velocity.t")
        assert pred_v.shape == x_t.shape, "pred_v and x_t shape mismatch"
        t_view = _broadcast_time(t, x_t.shape).to(device=x_t.device, dtype=x_t.dtype)
        return x_t + (1.0 - t_view) * pred_v

    @torch.no_grad()
    def euler_sample_loop(
        self,
        denoiser: nn.Module,
        shape: torch.Size,
        device: torch.device,
        n_steps: int | None = None,
        h_tokens: torch.Tensor | None = None,
        h_global: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Integrate dz/dt = v_theta(z, t, h) from noise to clean latent."""
        assert len(shape) == 4, "shape must be [B,T,J,D]"
        if h_tokens is not None:
            assert_shape(h_tokens, [shape[0], shape[1], None], "FlowMatchingProcess.euler_sample_loop.h_tokens")
        if h_global is not None:
            assert_shape(h_global, [shape[0], None], "FlowMatchingProcess.euler_sample_loop.h_global")

        steps = int(n_steps or self.ode_steps)
        if steps <= 0:
            raise ValueError("n_steps must be positive.")
        z = torch.randn(shape, device=device, generator=generator)
        dt = 1.0 / float(steps)

        use_cfg = float(cfg_scale) != 1.0 and (h_tokens is not None or h_global is not None)
        null_tokens = torch.zeros_like(h_tokens) if h_tokens is not None else None
        null_global = torch.zeros_like(h_global) if h_global is not None else None

        for i in range(steps):
            t = torch.full((shape[0],), i / float(steps), device=device, dtype=z.dtype)
            v_cond = denoiser(z, t, h_tokens=h_tokens, h_global=h_global)
            if use_cfg:
                v_null = denoiser(z, t, h_tokens=null_tokens, h_global=null_global)
                v = v_null + float(cfg_scale) * (v_cond - v_null)
            else:
                v = v_cond
            z = z + dt * v

        assert_shape(z, [shape[0], shape[1], shape[2], shape[3]], "FlowMatchingProcess.euler_sample_loop.z")
        return z

