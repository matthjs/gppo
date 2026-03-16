"""
CLEAR Replay Buffer with Reservoir Sampling.

Stores all information required for V-Trace / CLEAR:
  - observations
  - actions
  - rewards
  - dones
  - log-probabilities under the *behaviour* policy μ  (stored as log_probs)
  - value estimates V(s) from the behaviour policy network
  - action logits from the behaviour policy (needed for behavioral-cloning loss)

Reservoir sampling ensures that at any point the buffer holds a *uniform
random sample* of all experience seen so far, which is the approach used in
the paper (Appendix A.3).  Each unroll / transition is associated with a
random key in [0, 1].  A running threshold is maintained so that exactly
`capacity` items are kept above the threshold.
"""
from __future__ import annotations
import numpy as np
import torch
import torch
import numpy as np
from typing import Tuple

from gppo.util.rolloutbuffer import RolloutBuffer


def _to_tensor(x, dtype, device):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(np.asarray(x), dtype=dtype, device=device)


class ClearReplayBuffer:
    """
    Fixed-capacity replay buffer with reservoir sampling (Vitter's Algorithm R).

    Stores:
        obs       : [capacity, *obs_shape]          s_t
        next_obs  : [capacity, *obs_shape]          s_{t+1}  ← V-trace bootstrap
        actions   : [capacity, *action_shape]
        rewards   : [capacity]
        dones     : [capacity]
        log_probs : [capacity]   log μ(a|s)
        values    : [capacity]   V_μ(s)             ← BC value loss only
        logits    : [capacity, action_dim]           ← BC KL loss, discrete only
    """

    def __init__(
        self,
        capacity:     int,
        obs_shape:    Tuple[int, ...],
        action_shape: Tuple[int, ...],
        action_dim:   int,
        discrete:     bool,              # ← added
        device:       torch.device,
    ):
        self.capacity   = capacity
        self.device     = device
        self.action_dim = action_dim
        self.discrete   = discrete       # ← added

        self._size:       int = 0
        self._total_seen: int = 0

        self.obs      = torch.zeros((capacity, *obs_shape),    dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, *obs_shape),    dtype=torch.float32, device=device)
        # Discrete: store a single integer index [capacity, 1], matching RolloutBuffer
        # Continuous: store the full action vector [capacity, *action_shape]
        _action_shape = (1,) if discrete else action_shape
        self.actions  = torch.zeros((capacity, *_action_shape), dtype=torch.float32, device=device)
        self.rewards  = torch.zeros(capacity,                   dtype=torch.float32, device=device)
        self.dones    = torch.zeros(capacity,                   dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(capacity,                  dtype=torch.float32, device=device)
        self.values   = torch.zeros(capacity,                   dtype=torch.float32, device=device)
        self.logits   = torch.zeros((capacity, max(action_dim, 1)),
                                    dtype=torch.float32, device=device)

    def push(self, obs, next_obs, action, next_obs_=None,
                reward=None, done=None, log_prob=None, value=None, logits=None):
            obs_t    = _to_tensor(obs,      torch.float32, self.device)
            nobs_t   = _to_tensor(next_obs, torch.float32, self.device)
            rew_t    = _to_tensor(reward,   torch.float32, self.device).flatten()
            done_t   = _to_tensor(done,     torch.float32, self.device).flatten()
            lp_t     = _to_tensor(log_prob, torch.float32, self.device).flatten()
            val_t    = _to_tensor(value,    torch.float32, self.device).flatten()

            act_t = _to_tensor(action, torch.float32, self.device)
            if self.discrete:
                # Flatten to [n, 1] — matches RolloutBuffer convention
                act_t = act_t.view(-1, 1)
            
            n = rew_t.shape[0]

            if logits is not None:
                lg_t = _to_tensor(logits, torch.float32, self.device)
                if lg_t.dim() == 1:
                    lg_t = lg_t.unsqueeze(0).expand(n, -1)
            else:
                lg_t = torch.zeros((n, max(self.action_dim, 1)),
                                dtype=torch.float32, device=self.device)

            for i in range(n):
                self._push_single(
                    obs_t[i]  if obs_t.dim()  > 1 else obs_t,
                    nobs_t[i] if nobs_t.dim() > 1 else nobs_t,
                    act_t[i],   # always [1] for discrete, [action_dim] for continuous
                    rew_t[i], done_t[i], lp_t[i], val_t[i], lg_t[i],
                )

    def sample(self, batch_size: int):
            assert self._size > 0, "Cannot sample from an empty buffer."
            idx = torch.randint(0, self._size,
                                (min(batch_size, self._size),),
                                device=self.device)
            actions = self.actions[idx].long() if self.discrete else self.actions[idx]
            return (
                self.obs[idx],
                self.next_obs[idx],
                actions,               # [B, 1] long for discrete, [B, act_dim] float for continuous
                self.log_probs[idx],
                self.values[idx],
                self.logits[idx],
                self.rewards[idx],
                self.dones[idx],
            )

    def __len__(self) -> int:
        return self._size

    def _push_single(self, obs, next_obs, action, reward, done,
                     log_prob, value, logits):
        self._total_seen += 1

        if self._size < self.capacity:
            idx = self._size
            self._size += 1
        else:
            j = np.random.randint(0, self._total_seen)
            if j >= self.capacity:
                return
            idx = j

        self.obs[idx]       = obs
        self.next_obs[idx]  = next_obs  # ← added
        self.actions[idx]   = action
        self.rewards[idx]   = reward
        self.dones[idx]     = done
        self.log_probs[idx] = log_prob
        self.values[idx]    = value
        self.logits[idx]    = logits


def dump_rollout_to_replay(
    rollout,
    replay,
) -> None:
    """
    Transfer all stored transitions from a RolloutBuffer into a ClearReplayBuffer.

    Observations are shifted by one to produce (obs, next_obs) pairs.
    The final next_obs is a zero tensor (terminal / unknown bootstrap state).

    :param rollout: A filled RolloutBuffer (rollout.pos > 0).
    :param replay: The destination ClearReplayBuffer.
    :param action_dim: Number of discrete actions (for logits; pass 0 for continuous).
    """
    n = rollout.pos  # number of stored transitions

    obs     = rollout.obs[:n]           # [n, *obs_shape]
    actions = rollout.actions[:n]       # [n, *action_shape]
    rewards = rollout.rewards[:n]       # [n]
    dones   = rollout.dones[:n]         # [n]
    values  = rollout.values[:n]        # [n]
    log_probs = rollout.log_probs[:n]   # [n]

    # Build next_obs by shifting observations forward by one step.
    # For the final step, next_obs is zeroed (terminal / unavailable).
    next_obs = torch.zeros_like(obs)
    if n > 1:
        next_obs[:-1] = obs[1:]         # next_obs[t] = obs[t+1]
    # next_obs[-1] stays zero (end of rollout)

    # ClearReplayBuffer expects logits; on-policy rollouts don't store them,
    # so we pass None and let the buffer zero-fill the slot.
    replay.push(
        obs=obs,
        next_obs=next_obs,
        action=actions,
        reward=rewards,
        done=dones,
        log_prob=log_probs,
        value=values,
        logits=None,         # no logits stored in the rollout buffer
    )

"""
if __name__ == "__main__":
    import numpy as np
    import torch

    buf = ClearReplayBuffer(
        capacity     = 5,
        obs_shape    = (4,),
        action_shape = (1,),
        action_dim   = 2,          # discrete, 2 actions
        device       = torch.device("cpu"),
    )

    # --- push transitions one at a time ---
    for step in range(8):
        buf.push(
            obs      = np.random.randn(4).astype(np.float32),
            next_obs = np.random.randn(4).astype(np.float32),
            action   = np.array([np.random.randint(2)], dtype=np.float32),
            reward   = float(np.random.randn()),
            done     = step == 7,          # terminal on last step
            log_prob = np.float32(-0.693), # log(0.5), uniform policy
            value    = np.float32(np.random.randn()),
            logits   = np.random.randn(2).astype(np.float32),
        )
        print(f"step={step+1}  size={len(buf)}  total_seen={buf._total_seen}")

    # step=1  size=1  total_seen=1
    # step=2  size=2  total_seen=2
    # step=3  size=3  total_seen=3
    # step=4  size=4  total_seen=4
    # step=5  size=5  total_seen=5   ← buffer full from here
    # step=6  size=5  total_seen=6   ← reservoir sampling starts
    # step=7  size=5  total_seen=7
    # step=8  size=5  total_seen=8

    # --- sample a minibatch ---
    obs_b, next_obs_b, actions_b, log_probs_b, values_b, logits_b, rewards_b, dones_b \
        = buf.sample(batch_size=3)

    print(obs_b.shape)       # [3, 4]
    print(next_obs_b.shape)  # [3, 4]
    print(log_probs_b)       # log μ(a|s) — the behaviour policy's log probs
    print(values_b)          # V_μ(s)    — used only for BC value loss
"""
    
if __name__ == "__main__":
    """
    demo_transfer.py — shows that dump_rollout_to_replay works correctly.
    """
    import torch
    import numpy as np

    # ── config ────────────────────────────────────────────────────────────────────
    OBS_SHAPE    = (8,)
    ACTION_SHAPE = (2,)
    ACTION_DIM   = 0        # continuous actions → no logit storage needed
    N_ENVS       = 4
    T_STEPS      = 5        # timesteps per rollout
    CAPACITY     = 200
    DEVICE       = torch.device("cpu")
    GAMMA        = 0.99
    GAE_LAMBDA   = 0.95

    # ── build buffers ─────────────────────────────────────────────────────────────
    rollout = RolloutBuffer(
        capacity=N_ENVS * T_STEPS,
        obs_shape=OBS_SHAPE,
        action_shape=ACTION_SHAPE,
        device=DEVICE,
    )

    replay = ClearReplayBuffer(
        capacity=CAPACITY,
        obs_shape=OBS_SHAPE,
        action_shape=ACTION_SHAPE,
        action_dim=ACTION_DIM,
        device=DEVICE,
    )

    # ── fill rollout with fake transitions ───────────────────────────────────────
    obs = torch.randn(N_ENVS, *OBS_SHAPE)

    for t in range(T_STEPS):
        action   = torch.randn(N_ENVS, *ACTION_SHAPE)
        reward   = torch.rand(N_ENVS)
        done     = (torch.rand(N_ENVS) < 0.1).float()
        log_prob = torch.randn(N_ENVS)
        value    = torch.rand(N_ENVS)

        rollout.push(obs, action, reward, done, log_prob, value)
        obs = torch.randn(N_ENVS, *OBS_SHAPE)   # next observation

    last_value = torch.rand(N_ENVS)
    last_done  = torch.zeros(N_ENVS)
    rollout.compute_returns_and_advantages(last_value, last_done, GAMMA, GAE_LAMBDA)

    # ── transfer ──────────────────────────────────────────────────────────────────
    assert len(replay) == 0, "replay should be empty before transfer"

    dump_rollout_to_replay(rollout, replay)

    n_transferred = N_ENVS * T_STEPS   # 20
    assert len(replay) == n_transferred, f"expected {n_transferred}, got {len(replay)}"

    # ── verify a sample ───────────────────────────────────────────────────────────
    obs_s, next_obs_s, actions_s, log_probs_s, values_s, logits_s, rewards_s, dones_s = \
        replay.sample(batch_size=8)

    print("=== transfer demo ===")
    print(f"  rollout transitions : {rollout.pos}")
    print(f"  replay size after   : {len(replay)}")
    print(f"  sample obs shape    : {obs_s.shape}")
    print(f"  sample actions shape: {actions_s.shape}")
    print(f"  sample rewards      : {rewards_s}")
    print(f"  sample dones        : {dones_s}")

    # Spot-check: next_obs for non-terminal transitions should differ from obs
    non_terminal_mask = (dones_s == 0)
    if non_terminal_mask.any():
        diff = (obs_s[non_terminal_mask] - next_obs_s[non_terminal_mask]).abs().sum()
        assert diff > 0, "next_obs should differ from obs for non-terminal transitions"

    print("  all checks passed ✓")
