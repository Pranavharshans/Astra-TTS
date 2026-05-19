#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Union

import torch


class DiffusionModel(torch.nn.Module):
    """A wrapper of diffusion models for inference.
    Args:
        model: The diffusion model.
        func_name: The function name to call.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        func_name: str = "forward_fm_decoder",
    ):
        super().__init__()
        self.model = model
        self.func_name = func_name
        self.model_func = getattr(self.model, func_name)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Union[float, torch.Tensor] = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward function that Handles the classifier-free guidance.
        Args:
            t: The current timestep, a tensor of a tensor of a single float.
            x: The initial value, with the shape (batch, seq_len, emb_dim).
            text_condition: The text_condition of the diffision model, with
                the shape (batch, seq_len, emb_dim).
            speech_condition: The speech_condition of the diffision model, with the
                shape (batch, seq_len, emb_dim).
            padding_mask: The mask for padding; True means masked position, with the
                shape (batch, seq_len).
            guidance_scale: The scale of classifier-free guidance, a float or a tensor
                of shape (batch, 1, 1).
        Retrun:
            The prediction with the shape (batch, seq_len, emb_dim).
        """
        if not torch.is_tensor(guidance_scale):
            guidance_scale = torch.tensor(
                guidance_scale, dtype=t.dtype, device=t.device
            )

        if (guidance_scale == 0.0).all():
            return self.model_func(
                t=t,
                xt=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                **kwargs
            )
        else:
            assert t.dim() == 0

            x = torch.cat([x] * 2, dim=0)
            padding_mask = torch.cat([padding_mask] * 2, dim=0)

            text_condition = torch.cat(
                [torch.zeros_like(text_condition), text_condition], dim=0
            )

            if t > 0.5:
                speech_condition = torch.cat(
                    [torch.zeros_like(speech_condition), speech_condition], dim=0
                )
            else:
                guidance_scale = guidance_scale * 2
                speech_condition = torch.cat(
                    [speech_condition, speech_condition], dim=0
                )

            data_uncond, data_cond = self.model_func(
                t=t,
                xt=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                **kwargs
            ).chunk(2, dim=0)

            res = (1 + guidance_scale) * data_cond - guidance_scale * data_uncond
            return res


class DistillDiffusionModel(DiffusionModel):
    """A wrapper of distilled diffusion models for inference.
    Args:
        model: The distilled diffusion model.
        func_name: The function name to call.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        func_name: str = "forward_fm_decoder",
    ):
        super().__init__(model=model, func_name=func_name)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Union[float, torch.Tensor] = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward function that Handles the classifier-free guidance.
        Args:
            t: The current timestep, a tensor of a single float.
            x: The initial value, with the shape (batch, seq_len, emb_dim).
            text_condition: The text_condition of the diffision model, with
                the shape (batch, seq_len, emb_dim).
            speech_condition: The speech_condition of the diffision model, with the
                shape (batch, seq_len, emb_dim).
            padding_mask: The mask for padding; True means masked position, with the
                shape (batch, seq_len).
            guidance_scale: The scale of classifier-free guidance, a float or a tensor
                of shape (batch, 1, 1).
        Retrun:
            The prediction with the shape (batch, seq_len, emb_dim).
        """
        if not torch.is_tensor(guidance_scale):
            guidance_scale = torch.tensor(
                guidance_scale, dtype=t.dtype, device=t.device
            )
        return self.model_func(
            t=t,
            xt=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            guidance_scale=guidance_scale,
            **kwargs
        )


class EulerSolver:
    def __init__(
        self,
        model: torch.nn.Module,
        func_name: str = "forward_fm_decoder",
    ):
        """Construct a Euler Solver
        Args:
            model: The diffusion model.
            func_name: The function name to call.
        """

        self.model = DiffusionModel(model, func_name=func_name)

    def sample(
        self,
        x: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: torch.Tensor,
        num_step: int = 10,
        guidance_scale: Union[float, torch.Tensor] = 0.0,
        t_start: float = 0.0,
        t_end: float = 1.0,
        t_shift: float = 1.0,
        solver: str = "euler",
        step_schedule: str = "uniform",
        smooth_cache: bool = False,
        smooth_cache_stacks: Sequence[int] = (0, 1),
        smooth_cache_interval: int = 2,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute the sample at time `t_end` by Euler Solver.
        Args:
            x: The initial value at time `t_start`, with the shape (batch, seq_len,
                emb_dim).
            text_condition: The text condition of the diffision mode, with the
                shape (batch, seq_len, emb_dim).
            speech_condition: The speech condition of the diffision model, with the
                shape (batch, seq_len, emb_dim).
            padding_mask: The mask for padding; True means masked position, with the
                shape (batch, seq_len).
            num_step: The number of ODE steps.
            guidance_scale: The scale for classifier-free guidance, which is
                a float or a tensor with the shape (batch, 1, 1).
            t_start: the start timestep in the range of [0, 1].
            t_end: the end time_step in the range of [0, 1].
            t_shift: shift the t toward smaller numbers so that the sampling
                will emphasize low SNR region. Should be in the range of (0, 1].
                The shifting will be more significant when the number is smaller.

        Returns:
            The approximated solution at time `t_end`.
        """
        device = x.device
        assert isinstance(t_start, float) and isinstance(t_end, float)

        assert solver in ("euler", "midpoint"), solver
        assert smooth_cache_interval > 0, smooth_cache_interval

        timesteps = get_time_steps(
            t_start=t_start,
            t_end=t_end,
            num_step=num_step,
            t_shift=t_shift,
            step_schedule=step_schedule,
            device=device,
        )

        cache_state = {
            "enabled": smooth_cache,
            "stacks": tuple(smooth_cache_stacks),
            "values": {},
        }
        eval_idx = 0

        def model_eval(t: torch.Tensor, cur_x: torch.Tensor) -> torch.Tensor:
            nonlocal eval_idx
            if smooth_cache:
                cache_state["reuse"] = eval_idx % smooth_cache_interval != 0
                cache_state["eval_idx"] = eval_idx
                kwargs["smooth_cache_state"] = cache_state
            else:
                kwargs.pop("smooth_cache_state", None)
            eval_idx += 1
            return self.model(
                t=t,
                x=cur_x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
                **kwargs
            )

        for step in range(num_step):
            t0 = timesteps[step]
            t1 = timesteps[step + 1]
            dt = t1 - t0
            if solver == "euler":
                v = model_eval(t0, x)
                x = x + v * dt
            else:
                v0 = model_eval(t0, x)
                t_mid = t0 + dt * 0.5
                x_mid = x + v0 * dt * 0.5
                v_mid = model_eval(t_mid, x_mid)
                x = x + v_mid * dt
        return x


class DistillEulerSolver(EulerSolver):
    def __init__(
        self,
        model: torch.nn.Module,
        func_name: str = "forward_fm_decoder",
    ):
        """Construct a Euler Solver for distilled diffusion models.
        Args:
            model: The diffusion model.
        """
        self.model = DistillDiffusionModel(model, func_name=func_name)


def get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
    step_schedule: str = "uniform",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute the intermediate time steps for sampling.

    Args:
        t_start: The starting time of the sampling (default is 0).
        t_end: The starting time of the sampling (default is 1).
        num_step: The number of sampling.
        t_shift: shift the t toward smaller numbers so that the sampling
            will emphasize low SNR region. Should be in the range of (0, 1].
            The shifting will be more significant when the number is smaller.
        device: A torch device.
    Returns:
        The time step with the shape (num_step + 1,).
    """

    assert step_schedule in ("uniform", "epss"), step_schedule

    if step_schedule == "uniform":
        timesteps = torch.linspace(t_start, t_end, num_step + 1).to(device)
    else:
        base = torch.tensor(
            [0.0, 0.05, 0.18, 0.38, 0.62, 0.82, 0.95, 1.0], device=device
        )
        if num_step + 1 == base.numel():
            timesteps = base
        else:
            base_x = torch.linspace(0.0, 1.0, base.numel(), device=device)
            target_x = torch.linspace(0.0, 1.0, num_step + 1, device=device)
            indexes = torch.searchsorted(base_x, target_x, right=True).clamp(
                1, base.numel() - 1
            )
            left_x = base_x[indexes - 1]
            right_x = base_x[indexes]
            left_y = base[indexes - 1]
            right_y = base[indexes]
            weight = (target_x - left_x) / (right_x - left_x)
            timesteps = left_y + weight * (right_y - left_y)
        timesteps = t_start + (t_end - t_start) * timesteps

    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)

    return timesteps
