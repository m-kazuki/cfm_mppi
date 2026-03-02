from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

class FlowMPPI(nn.Module):
    def __init__(
        self,
        num_samples: int,
        dim_state: int,
        dim_control: int,
        dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        stage_cost: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        terminal_cost: Callable[[torch.Tensor], torch.Tensor],
        u_min: torch.Tensor,
        u_max: torch.Tensor,
        sigmas: torch.Tensor,
        lambda_: float,
        goal: torch.Tensor,
        horizon: int,
        dt: float,
        device=torch.device("cuda"),
        dtype=torch.float32,
        seed: int = 42,
        dynamics_type: str = "unicycle",
    ) -> None:
        """
        :param delta: predictive horizon step size (seconds).
        :param num_samples: Number of samples.
        :param dim_state: Dimension of state.
        :param dim_control: Dimension of control.
        :param dynamics: Dynamics model.
        :param stage_cost: Stage cost.
        :param terminal_cost: Terminal cost.
        :param u_min: Minimum control.
        :param u_max: Maximum control.
        :param sigmas: Noise standard deviation for each control dimension.
        :param lambda_: temperature parameter.
        :param device: Device to run the solver.
        :param dtype: Data type to run the solver.
        :param seed: Seed for torch.
        """

        super().__init__()

        # check dimensions
        assert u_min.shape == (dim_control,)
        assert u_max.shape == (dim_control,)
        assert sigmas.shape == (dim_control,)

        # device and dtype
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype

        # set parameters
        self._num_samples = num_samples
        self._dim_state = dim_state
        self._dim_control = dim_control
        self._dynamics = dynamics
        self._stage_cost = stage_cost
        self._terminal_cost = terminal_cost
        self._u_min = u_min.clone().detach().to(self._device, self._dtype)
        self._u_max = u_max.clone().detach().to(self._device, self._dtype)
        self._sigmas = sigmas.clone().detach().to(self._device, self._dtype)
        self._lambda = lambda_

        # noise distribution
        zero_mean = torch.zeros(dim_control, device=self._device, dtype=self._dtype)
        initial_covariance = torch.diag(sigmas**2).to(self._device, self._dtype)
        self._inv_covariance = torch.inverse(initial_covariance).to(
            self._device, self._dtype
        )

        self._noise_distribution = MultivariateNormal(
            loc=zero_mean, covariance_matrix=initial_covariance
        )

        # dynamics type
        assert dynamics_type in ("unicycle", "doubleintegrator", "singleintegrator"), \
            f"Unknown dynamics_type: {dynamics_type}"
        self._dynamics_type = dynamics_type

        self._dt = dt

        self.prev_optimal_action_seq = None

    def _convert_si_to_dynamics(self, controls_sin_t, state_t, d, controls_sin_t_next=None, k_p=2.0):
        """
        Convert single integrator controls to the target dynamics controls.
        Args:
            controls_sin_t (torch.Tensor): SI controls at time t, shape (num_samples, 2) [vx, vy]
            state_t (torch.Tensor): state at time t, shape (num_samples, dim_state)
            d (float): look-ahead distance for unicycle
            controls_sin_t_next (torch.Tensor, optional): SI controls at time t+1 for feedforward (DI only)
            k_p (float): feedback gain for double integrator velocity tracking
        Returns:
            torch.Tensor: converted controls, shape (num_samples, 2)
        """
        if self._dynamics_type == "unicycle":
            # SI [vx, vy] -> unicycle [v, omega] via look-ahead control
            return torch.cat(
                [
                    controls_sin_t[:, 0:1] * torch.cos(state_t[:, 2:3]) + controls_sin_t[:, 1:2] * torch.sin(state_t[:, 2:3]),
                    (-1/d) * controls_sin_t[:, 0:1] * torch.sin(state_t[:, 2:3]) + (1/d) * controls_sin_t[:, 1:2] * torch.cos(state_t[:, 2:3]),
                ], dim=-1
            )
        elif self._dynamics_type == "doubleintegrator":
            # SI [vx, vy] -> DI [ax, ay] via feedforward + feedback
            # Feedforward: desired acceleration from SI velocity trajectory
            if controls_sin_t_next is not None:
                a_ff = (controls_sin_t_next - controls_sin_t) / self._dt
            else:
                a_ff = torch.zeros_like(controls_sin_t)
            # Feedback: proportional velocity tracking
            a_fb = torch.cat(
                [
                    k_p * (controls_sin_t[:, 0:1] - state_t[:, 2:3]),
                    k_p * (controls_sin_t[:, 1:2] - state_t[:, 3:4]),
                ], dim=-1
            )
            return a_ff + a_fb
        elif self._dynamics_type == "singleintegrator":
            # No conversion needed
            return controls_sin_t

    def cost_func_for_mode(self, state, control, obstacle, prev_action, time, goal, rad):
        return self._stage_cost(
            state, control, goal, obstacle, rad, time=time, prev_action=prev_action
        )
    
    def cost_func_for_mppi(self, state, control, obstacle, prev_action, mean_action, time, goal, rad):
        stage_cost = self._stage_cost(
            state, control, goal, obstacle, rad, time=time, prev_action=prev_action
        )
        action_cost = mean_action @ self._inv_covariance @ control.T
        return stage_cost, action_cost

    def forward(self, state, controls_sin, horizon, goal, obstacle_state, rad, d=0.1, k_p=2.0) -> torch.Tensor:
        """
        Solve the optimal control problem.
        Args:
            state (torch.Tensor): Current state [3].
            controls (torch.Tensor): Control sequence [num_sample, horizon, 2].
        Returns:
            optimal controls (torch.Tensor): Optimal control sequence [horizon, 2].
        """

        if not torch.is_tensor(state):
            state = torch.tensor(state, device=self._device, dtype=self._dtype)
        else:
            if state.device != self._device or state.dtype != self._dtype:
                state = state.to(self._device, self._dtype)
        
        # rollout samples in parallel
        state_seq_batch = torch.zeros(self._num_samples, horizon + 1, self._dim_state).to(state.device)
        state_seq_batch[:, 0, :] = state.repeat(self._num_samples, 1)
        controls_dyn = torch.zeros_like(controls_sin).to(self._device)

        for t in range(horizon):
            controls_sin_t_next = controls_sin[:, t + 1, :] if t < horizon - 1 else None
            control_dyn = self._convert_si_to_dynamics(
                controls_sin[:, t, :], state_seq_batch[:, t, :], d,
                controls_sin_t_next=controls_sin_t_next, k_p=k_p
            ) # [num_samples, 2]

            control_dyn = torch.clamp(
                control_dyn, self._u_min, self._u_max
            )

            controls_dyn[:, t, :] = control_dyn
            state_seq_batch[:, t + 1, :] = self._dynamics(
                state_seq_batch[:, t, :], control_dyn
            )

        initial_prev_action = torch.zeros_like(controls_dyn[:, :1, :], device=state_seq_batch.device)
        prev_actions = torch.cat([initial_prev_action, controls_dyn[:, :-1, :]], dim=1)
        times = torch.arange(horizon, device=state_seq_batch.device)
        vectorized_cost = torch.vmap(
            self.cost_func_for_mode,
            in_dims=(1, 1, 1, 1, 0, None, None),
            out_dims=1
        )
        stage_costs = vectorized_cost(
            state_seq_batch[:,:-1,:], 
            controls_dyn, 
            obstacle_state, 
            prev_actions,
            times,
            goal, 
            rad
        )

        terminal_costs = self._terminal_cost(state_seq_batch[:, -1, :2], goal)

        costs = (
            torch.sum(stage_costs, dim=1)
            + terminal_costs
        )
        if self.prev_optimal_action_seq is not None:
            costs += 0.1 * torch.sum(
                (self.prev_optimal_action_seq.unsqueeze(0)[:,1:,:] - controls_dyn) ** 2, dim=(1, 2)
            )

        n_elite = 10
        n_copy = self._num_samples # max(self._num_samples // n_elite, 1)
        n_batch = n_elite * n_copy

        _, topk_indices = torch.topk(costs, k=n_elite, largest=False, dim=0)
        mean_action_seq = controls_dyn[topk_indices]  # [n_elite, horizon, dim_control]
        mean_action_seq_sin = controls_sin[topk_indices]  # [n_elite, horizon, dim_control]
        mean_action_seq = mean_action_seq.repeat_interleave(n_copy, dim=0) # [n_sample, horizon, dim_control]


        # random sampling with reparametrization trick
        action_noises = self._noise_distribution.rsample(
            sample_shape=torch.Size([n_batch, horizon])
        ) # [num_samples, horizon, dim_control]


        perturbed_action_seqs = mean_action_seq + action_noises

        # clamp actions
        perturbed_action_seqs = torch.clamp(
            perturbed_action_seqs, self._u_min, self._u_max
        )

        # rollout samples in parallel
        state_seq_batch = torch.zeros(n_batch, horizon + 1, self._dim_state).to(state.device)
        state_seq_batch[:, 0, :] = state.repeat(n_batch, 1)

        for t in range(horizon):
            state_seq_batch[:, t + 1, :] = self._dynamics(
                state_seq_batch[:, t, :], perturbed_action_seqs[:, t, :]
            )


        initial_prev_action = torch.zeros_like(perturbed_action_seqs[:, :1, :])
        prev_actions = torch.cat([initial_prev_action, perturbed_action_seqs[:, :-1, :]], dim=1)
        times = torch.arange(horizon, device=self._device)

        vectorized_costs = torch.vmap(
            self.cost_func_for_mppi,
            in_dims=(1, 1, 1, 1, 1, 0, None, None),
            out_dims=1
        )
        stage_costs, action_costs = vectorized_costs(
            state_seq_batch[:,:-1,:], 
            perturbed_action_seqs,
            obstacle_state,
            prev_actions,
            mean_action_seq,
            times,
            goal,
            rad
        )

        terminal_costs = self._terminal_cost(state_seq_batch[:, -1, :2], goal)

        costs = (
            torch.sum(stage_costs, dim=1)
            + terminal_costs
        ) # [n_sample]

        reshaped_cost = costs.reshape(n_elite, n_copy)
        betas, _ = torch.min(reshaped_cost, dim=1) # [n_elite]

        weights = torch.softmax(-(reshaped_cost - betas.view(n_elite,1)) / self._lambda, dim=1) # [n_elite, n_copy]
        optimal_action_seq = torch.sum(
            weights.view(n_elite, n_copy, 1, 1) * perturbed_action_seqs.view(n_elite, n_copy, horizon, self._dim_control),
            dim=1,
        ) # [n_elite, horizon, dim_control]

        state_seq_batch = torch.zeros(n_elite, horizon + 1, self._dim_state).to(state.device)
        state_seq_batch[:, 0, :] = state.repeat(n_elite, 1)

        for t in range(horizon):
            state_seq_batch[:, t + 1, :] = self._dynamics(
                state_seq_batch[:, t, :], optimal_action_seq[:, t, :]
            )
        initial_prev_action = torch.zeros_like(optimal_action_seq[:, :1, :], device=state_seq_batch.device)
        prev_actions = torch.cat([initial_prev_action, optimal_action_seq[:, :-1, :]], dim=1)
        times = torch.arange(horizon, device=state_seq_batch.device)
        vectorized_cost = torch.vmap(
            self.cost_func_for_mode,
            in_dims=(1, 1, 1, 1, 0, None, None),
            out_dims=1
        )
        stage_costs = vectorized_cost(
            state_seq_batch[:,:-1,:], 
            optimal_action_seq, 
            obstacle_state, 
            prev_actions,
            times,
            goal, 
            rad
        )
        terminal_costs = self._terminal_cost(state_seq_batch[:, -1, :2], goal)
        costs = (
            torch.sum(stage_costs, dim=1)
            + terminal_costs
        )
        _, idx = torch.min(costs, dim=0)
        optimal_action_seq = optimal_action_seq[idx]  # [horizon, dim_control]
        mean_action_seq_sin = mean_action_seq_sin[idx]  # [horizon, dim_control]

        self.prev_optimal_action_seq = optimal_action_seq

        return optimal_action_seq, mean_action_seq_sin


    def get_top_samples(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top samples.
        Args:
            num_samples (int): Number of state samples to get.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of top samples and their weights.
        """
        assert num_samples <= self._num_samples

        # large weights are better
        top_indices = torch.topk(self._weights, num_samples).indices

        top_samples = self._state_seq_batch[top_indices]
        top_weights = self._weights[top_indices]

        top_samples = top_samples[torch.argsort(top_weights, descending=True)]
        top_weights = top_weights[torch.argsort(top_weights, descending=True)]

        return top_samples, top_weights