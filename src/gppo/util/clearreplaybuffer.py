from __future__ import annotations
import numpy as np
import torch
import torch
import numpy as np
from typing import Tuple
from typing import Generator

from gppo.util.rolloutbuffer import RolloutBuffer

def _to_tensor(x, dtype, device):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(np.asarray(x), dtype=dtype, device=device)


class ClearReplayBuffer:
    """
    Fixed-capacity replay buffer that stores contiguous *unrolls* of length
    ``unroll_len`` rather than individual transitions.  Storing unrolls is
    required so that the full V-trace return (with c̄ trace coefficients) can
    be computed during the off-policy update — a single-step buffer makes
    ``c_bar`` a dead parameter because the trace product over t=s..s+n-1 is
    always the empty product (= 1) when n=1.

    Capacity is measured in **unrolls**, so the total number of environment
    frames stored is ``capacity * unroll_len``.

    Reservoir sampling (Vitter's Algorithm R) is used once the buffer is full,
    ensuring that at any point the buffer holds a uniformly random sample of
    all unrolls seen so far.

    Stored fields per unroll slot
    ──────────────────────────────
        obs          [unroll_len, *obs_shape]        s_0 … s_{T-1}
        actions      [unroll_len, *action_shape]
        rewards      [unroll_len]
        dones        [unroll_len]
        log_probs    [unroll_len]                    log μ(a_t | s_t)
        values       [unroll_len]                    V_μ(s_t)
        logits       [unroll_len, action_dim]        raw logits of μ  (discrete)
        next_obs     [unroll_len, *obs_shape]        s_{t+1} for each timestep
    """

    def __init__(
        self,
        capacity:     int,
        obs_shape:    Tuple[int, ...],
        action_shape: Tuple[int, ...],
        unroll_len:   int,
        n_envs:       int,
        action_dim:   int,           # number of discrete actions; 0 for continuous
        discrete:     bool,
        device:       torch.device,
    ):
        """
        :param capacity: Maximum number of unrolls to store.
        :param obs_shape: Shape of a single observation.
        :param action_shape: Shape of a single action.
        :param unroll_len: Number of timesteps per stored unroll (T).
        :param action_dim: Number of discrete actions.  Pass 0 for continuous
            spaces — the logits array is still allocated but unused.
        :param discrete: Whether the action space is discrete.
        :param device: Torch device for all stored tensors.
        """
        self.capacity   = capacity
        self.unroll_len = unroll_len
        self.device     = device
        self.action_dim = action_dim
        self.discrete   = discrete
        self.n_envs = n_envs

        self._size:       int = 0   # number of valid unroll slots
        self._total_seen: int = 0   # total unrolls pushed (for reservoir)

        # pre-allocate storage
        self.obs       = torch.zeros(
            (capacity, unroll_len, *    obs_shape),    dtype=torch.float32, device=device)
        self.next_obs = torch.zeros(
            (capacity, unroll_len, *obs_shape), dtype=torch.float32, device=device)
        _action_shape = (1,) if discrete else action_shape
        self.actions   = torch.zeros(
            (capacity, unroll_len, *_action_shape), dtype=torch.float32, device=device)

        self.rewards   = torch.zeros((capacity, unroll_len), dtype=torch.float32, device=device)
        self.dones     = torch.zeros((capacity, unroll_len), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((capacity, unroll_len), dtype=torch.float32, device=device)
        self.values    = torch.zeros((capacity, unroll_len), dtype=torch.float32, device=device)
        self.logits    = torch.zeros(
            (capacity, unroll_len, max(action_dim, 1)), dtype=torch.float32, device=device)


    def push(
        self,
        obs:           torch.Tensor,            # [T, *obs_shape]
        next_obs: torch.Tensor,                 # [T, *obs_shape]
        actions:       torch.Tensor,            # [T, *action_shape]
        rewards:       torch.Tensor,            # [T]
        dones:         torch.Tensor,            # [T]
        log_probs:     torch.Tensor,            # [T]
        values:        torch.Tensor,            # [T]
        logits:        torch.Tensor | None,     # [T, action_dim]  or None
    ) -> None:
        """
        Store one unroll.  All tensors must already have the leading T dimension.
        Reservoir sampling is applied automatically once the buffer is full.
        """
        assert obs.shape[0] == self.unroll_len, (
            f"Expected unroll of length {self.unroll_len}, got {obs.shape[0]}")

        self._total_seen += 1

        # Decide which slot to write into (reservoir sampling)
        if self._size < self.capacity:
            idx = self._size
            self._size += 1
        else:
            j = np.random.randint(0, self._total_seen)
            if j >= self.capacity:
                return          # discard — keeps buffer a uniform random sample
            idx = j

        self.obs[idx]       = _to_tensor(obs,      torch.float32, self.device)
        self.next_obs[idx]  = _to_tensor(next_obs, torch.float32, self.device)
        self.rewards[idx]   = _to_tensor(rewards,  torch.float32, self.device)
        self.dones[idx]     = _to_tensor(dones,    torch.float32, self.device)
        self.log_probs[idx] = _to_tensor(log_probs,torch.float32, self.device)
        self.values[idx]    = _to_tensor(values,   torch.float32, self.device)

        act = _to_tensor(actions, torch.float32, self.device)
        self.actions[idx] = act

        if logits is not None:
            self.logits[idx] = _to_tensor(logits, torch.float32, self.device)
        # else: logits slot stays zero (harmless for continuous; never used for KL)

    def sample(self, policy,
               gamma: float, gae_lambda: float, n_envs: int = 1,
               batch_size: int = 1) -> Generator:
        """
        Sample all stored unrolls, compute GAE, then yield mini-batches.
        Flattened over T*n_envs to match the RolloutBuffer.get() interface.

        :param gamma: Discount factor.
        :param gae_lambda: GAE lambda parameter.
        :param batch_size: Size of mini-batches to yield.
        :param n_envs: Number of parallel environments (for proper reshaping).

        Yields
        ------
        obs          [batch_size, *obs_shape]
        actions      [batch_size, *action_shape]
        log_probs    [batch_size, 1]
        returns      [batch_size, 1]
        advantages   [batch_size, 1]
        logits       [batch_size, action_dim]
        values_mu    [batch_size, 1]             stored V_μ(s) for bc_value_loss
        """
        assert self._size > 0, "Cannot sample from an empty buffer."

        idx = torch.randint(0, self._size, (1,), device=self.device)

        # Get data - stored as [capacity, unroll_len, ...]
        # After indexing: [1, unroll_len, ...]
        obs           = self.obs[idx]           # [B, unroll_len, *obs_shape]
        next_obs      = self.next_obs[idx]      # [B, unroll_len, *obs_shape]
        actions       = self.actions[idx]       # [B, unroll_len, *action_shape]
        rewards       = self.rewards[idx]       # [B, unroll_len]
        dones         = self.dones[idx]         # [B, unroll_len]
        values        = self.values[idx]        # [B, unroll_len]
        log_probs     = self.log_probs[idx]     # [B, unroll_len]
        logits        = self.logits[idx]        # [B, unroll_len, action_dim]

        if self.discrete:
            actions = actions.long()

        total_timesteps = self.unroll_len
        T = total_timesteps // n_envs

        def _flatten_unrolls(x):
            return x.squeeze(0)

        obs_flat       = _flatten_unrolls(obs)        # [T*n_envs, *obs_shape]
        next_obs_flat  = _flatten_unrolls(next_obs)   # [T*n_envs, *obs_shape]
        actions_flat   = _flatten_unrolls(actions)
        rewards_flat   = _flatten_unrolls(rewards)
        dones_flat     = _flatten_unrolls(dones)
        values_flat    = _flatten_unrolls(values)
        log_probs_flat = _flatten_unrolls(log_probs)
        logits_flat    = _flatten_unrolls(logits)

        # Now reshape into [T, n_envs] like RolloutBuffer does
        rewards_reshaped  = rewards_flat.view(T, n_envs)
        dones_reshaped    = dones_flat.view(T, n_envs)

        # Recompute values with current critic
        with torch.no_grad():
            _, values_current, _ = policy(obs_flat)       # [T*n_envs]
            _, next_values, _    = policy(next_obs_flat)  # [T*n_envs]

        values_reshaped      = values_current.view(T, n_envs)
        next_values_reshaped = next_values.view(T, n_envs)

        advantages = torch.zeros(T, n_envs, device=self.device)
        gae        = torch.zeros(n_envs, device=self.device)

        # Compute GAE: δt = rt + γV(st+1) - V(st)
        for t in reversed(range(T)):
            delta = (
                rewards_reshaped[t]
                + gamma * next_values_reshaped[t] * (1.0 - dones_reshaped[t])
                - values_reshaped[t]
            )
            gae = delta + gamma * gae_lambda * (1.0 - dones_reshaped[t]) * gae
            advantages[t] = gae

        returns = advantages + values_reshaped

        # Flatten back to match buffer layout
        advantages_flat_final = advantages.view(-1)
        returns_flat_final    = returns.view(-1)

        # Normalize advantages
        # advantages_flat_final = (
        #    (advantages_flat_final - advantages_flat_final.mean()) /
        #    (advantages_flat_final.std() + 1e-8)
        # )

        # Shuffle and yield mini-batches
        indices = torch.randperm(total_timesteps, device=self.device)
        for i in range(0, total_timesteps, batch_size):
            batch_idx = indices[i:i + batch_size]
            yield (
                obs_flat[batch_idx],
                actions_flat[batch_idx],
                log_probs_flat[batch_idx].unsqueeze(-1),
                returns_flat_final[batch_idx].unsqueeze(-1),
                advantages_flat_final[batch_idx].unsqueeze(-1),
                logits_flat[batch_idx] if self.discrete else None,
                values_flat[batch_idx].unsqueeze(-1),   # stored V_μ(s) for bc_value_loss
            )

    def __len__(self) -> int:
        return self._size


def dump_rollout_to_replay(
    rollout:  RolloutBuffer,
    replay:   ClearReplayBuffer,
) -> None:
    """
    Slice the completed ``RolloutBuffer`` into non-overlapping unrolls of
    length ``replay.unroll_len`` and push each one into the replay buffer.

    Observations are already stored in the rollout as s_0…s_{T-1}.  The
    bootstrap observation for each unroll (s_T, used by V-trace) is taken
    as the first observation of the *next* unroll, or zeros for the very
    last unroll (end-of-rollout boundary).

    Any trailing steps that do not fill a complete unroll are discarded —
    consistent with how the original IMPALA-based paper handles unrolls.

    :param rollout: A filled ``RolloutBuffer`` (``rollout.pos > 0``).
    :param replay: Destination ``ClearReplayBuffer``.
    """
    T   = replay.unroll_len
    n   = rollout.pos           # total steps stored (n_envs * n_steps)
    n_complete = (n // T) * T   # drop any incomplete trailing unroll

    if n_complete == 0:
        return

    obs       = rollout.obs[:n_complete]        # [n_complete, *obs_shape]
    actions   = rollout.actions[:n_complete]    # [n_complete, *act_shape]
    rewards   = rollout.rewards[:n_complete]    # [n_complete]
    dones     = rollout.dones[:n_complete]      # [n_complete]
    values    = rollout.values[:n_complete]     # [n_complete]
    log_probs = rollout.log_probs[:n_complete]  # [n_complete]
    logits    = rollout.logits[:n_complete] if rollout.logits is not None else None

    # derive next_obs by shifting obs by n_envs steps
    n_envs   = replay.n_envs
    next_obs = torch.cat([obs[n_envs:], torch.zeros_like(obs[:n_envs])], dim=0)  # [n_complete, *obs_shape]

    num_unrolls = n_complete // T

    for i in range(num_unrolls):
        start = i * T
        end   = start + T

        replay.push(
            obs       = obs[start:end],
            next_obs  = next_obs[start:end],
            actions   = actions[start:end],
            rewards   = rewards[start:end],
            dones     = dones[start:end],
            log_probs = log_probs[start:end],
            values    = values[start:end],
            logits    = logits[start:end] if logits is not None else None,
        )
