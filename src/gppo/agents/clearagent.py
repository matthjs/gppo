
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from torch.nn import functional as F
from gppo.agents.ppoagent import PPOAgent
from gppo.util.clearreplaybuffer import ClearReplayBuffer, dump_rollout_to_replay
import torch
import numpy as np
from typing import Optional
from itertools import cycle, repeat

class CLEARAgent(PPOAgent):
    """
    Implementation of CLEAR with support for parallel environment execution.
    Unlike the original implementation, the standard PPO surrogate loss is used
    for on-policy data and for off-policy replay data an IS-weighted GAE
    (can be) is used instead of the V-trace algorithm with a 1-step IS-weighted
    TD-error.

    Supports both continuous and discrete action spaces. For image-based observations
    (e.g. Minigrid), pass a `features_extractor_class` via `features_extractor_class`
    and `features_extractor_kwargs` to inject a custom CNN backbone, mirroring SB3's
    policy_kwargs pattern.
    """

    def __init__(
        self,
        # ── environment ────────────────────────────────────────────────
        state_dimensions,
        action_dimensions,
        # ── rollout / training ─────────────────────────────────────────
        n_steps:        int   = 2048,
        n_envs:         int   = 1,
        batch_size:     int   = 64,
        learning_rate:  float = 3e-4,
        n_epochs:       int   = 4,
        gamma:          float = 0.99,
        gae_lambda:     float = 0.95,
        # ── PPO clipping ───────────────────────────────────────────────
        clip_range:     float          = 0.2,
        clip_range_vf:  Optional[float]= None,
        ent_coef:       float = 0.0,
        vf_coef:        float = 0.5,
        max_grad_norm:  float = 0.5,
        # ── CLEAR-specific ─────────────────────────────────────────────
        replay_buffer_capacity: int   = 50_000,
        new_replay_ratio:       float = 0.5,   # fraction of batch from NEW data
        bc_policy_coef:         float = 0.01,  # weight for L_policy_cloning
        bc_value_coef:          float = 0.005, # weight for L_value_cloning
        rho_bar:                float = 1.0,   # IS clip for policy gradient
        # c_bar:                  float = 1.0,   # IS clip for trace (GAE recursion)
        # use_is_gae:             bool  = True,  # currently always done, not used
        use_ppo_loss:           bool = True,     # use PPO surrogate loss for on-policy data
        behavioral_cloning:     bool = True,
        # ── infrastructure ────────────────────────────────────────────
        device:         torch.device   = torch.device("cpu"),
        torch_compile:  bool           = False,
        optimizer_cfg:  dict           = None,
        features_extractor_class       = None,
        features_extractor_kwargs: dict= None,
        discrete:       bool           = False,
        **kwargs,
    ):
        """
        :param state_dimensions: Shape of the input state.
        :param action_dimensions: Shape of the actions.
        :param n_steps: Number of steps per environment before updating.
        :param n_envs: Number of parallel environments.
        :param batch_size: Minibatch size
        :param learning_rate: learning rate for SGD based optimizer
        :param n_epochs: Number of epoch when optimizing surrogate loss
        :param gamma: discount factor
        :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        :param clip_range: Clipping parameter, it can be a function of the remaining progress.
        :param clip_range_vf: Clipping parameter for the value function, None means no clipping
        will be performed for the value function.
        :param ent_coef: Entropy coefficient for the loss calculation (c_1).
        :param vf_coef: Value function coefficient for the loss calculation (c_2).
        :param max_grad_norm: The maximum value for the gradient clipping
        :param device: Device.
        :param torch_compile: Whether to compile the policy with torch.compile.
        :param optimizer_cfg: Optional optimizer config dict (resolved via resolve_optimizer_cls).
        :param features_extractor_class: Optional CNN feature extractor class. Must expose a
        `features_dim: int` attribute. When provided, ActorCriticCNN is used as the policy;
        otherwise ActorCriticMLP is used. Mirrors SB3's policy_kwargs["features_extractor_class"].
        :param features_extractor_kwargs: Keyword arguments passed to features_extractor_class.__init__.
        :param discrete: Whether the action space is discrete. Applies to both MLP and CNN policies:
        uses a Categorical distribution when True, Gaussian when False.
        """
        super().__init__(
            state_dimensions=state_dimensions,
            action_dimensions=action_dimensions,
            n_steps=n_steps,
            n_envs=n_envs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            device=device,
            torch_compile=torch_compile,
            optimizer_cfg=optimizer_cfg,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            discrete=discrete,
            kwargs=kwargs
        )

        # CLEAR params
        self.new_replay_ratio = float(new_replay_ratio)
        self.bc_policy_coef   = bc_policy_coef
        self.bc_value_coef    = bc_value_coef
        self.rho_bar          = rho_bar
        # self.c_bar            = c_bar
        # self.use_is_gae       = use_is_gae
        self.replay_buffer_capacity = replay_buffer_capacity
        self.use_ppo_loss = use_ppo_loss
        self.behavioral_cloning = behavioral_cloning

        # CLEAR replay buffer
        action_dim = int(np.prod(action_dimensions))
        # Logit management
        self.last_logits = None
        self.rollout_buffer.set_logit_buffer(action_dim)

        self.replay_buffer = ClearReplayBuffer(
            capacity=replay_buffer_capacity,
            obs_shape=state_dimensions,
            action_shape=action_dimensions,
            # Each unroll will be the amount in the replaybuffer before
            # clear
            unroll_len=self.rollout_buffer.capacity,   # n_envs * n_steps !
            n_envs=self.n_envs,
            action_dim=action_dim if discrete else 0,   # --> for later, means do not allocate logits array
            discrete=discrete,
            device=device,
        )

        if self.torch_compile:
            self._compute_clear_loss = torch.compile(self._compute_clear_loss)

    def choose_action(self, observation: np.ndarray) -> Any:
        self.policy.eval()
        with torch.no_grad():
            # Support both single and parallel environments
            state = torch.tensor(
                observation, dtype=torch.float32, device=self.device)    # [n_env, dim]
            dist, value, _ = self.policy(state)    # value: [n_env]
            action = dist.sample()    # [n_env, dim]
            # Categorical log_prob is already scalar per sample; Normal needs summing over action dims
            lp = dist.log_prob(action)
            log_prob = lp if lp.dim() == 1 else lp.sum(dim=-1)    # log_prob: [n_env]
            self.last_log_prob = log_prob
            self.last_value = value
            # Buffer expects [n_envs, action_dim]; Categorical samples [n_envs] so unsqueeze
            if self.discrete:
                action = action.unsqueeze(-1)
                self.last_logits = dist.logits    # <--- added
            return action.cpu().numpy()

    def store_transition(
            self,
            state: np.ndarray,
            action: Any,
            reward: float,
            new_state: np.ndarray,
            done: bool,
    ) -> None:
        self.next_state = new_state
        self.last_done = done
        self.rollout_buffer.push(
            state,
            action,
            reward,
            done,
            self.last_log_prob.detach(),
            self.last_value.detach(),
            self.last_logits.detach() if self.discrete else None,   # <--- added
            self.next_state
        )

    @staticmethod
    def _compute_clear_loss(
        replay_obs: torch.Tensor,
        replay_actions: torch.Tensor,
        replay_log_probs_mu: torch.Tensor,
        replay_returns: torch.Tensor,
        replay_advantages: torch.Tensor,
        replay_logits: torch.Tensor,
        replay_values_mu: torch.Tensor,

        policy,

        rho_bar: float,
        vf_coef: float,
        ent_coef: float,
        bc_policy_coef: float,
        bc_value_coef: float,
        discrete: bool,
        behavioral_cloning: bool = True
    ) -> tuple:
        # get forward pass
        dist_s, value_s, _ = policy(replay_obs)
        
        # not sure if I need to do this
        # [B] -> [B, 1]
        value_s = value_s.unsqueeze(-1)
        # [B, 1] -> [B] if needed
        _actions = replay_actions.squeeze(-1) if discrete else replay_actions

        log_prob = dist_s.log_prob(_actions)
        log_prob = (log_prob if log_prob.dim() == 1 else log_prob.sum(dim=-1)).unsqueeze(-1)

        # IS ratio  ρ = min(ρ̄, π_θ(a|s) / μ(a|s))
        # [B, 1]
        rho = torch.exp(log_prob - replay_log_probs_mu).clamp(max=rho_bar).detach()

        # policy gradient loss  −ρ · log π_θ(a|s) · advantage
        policy_gradient_loss = (-rho * log_prob * replay_advantages.detach()).mean()

        # Value loss (V_θ(s) − v_s)²
        value_loss = F.mse_loss(value_s, replay_returns)

        ent = dist_s.entropy()
        entropy_loss = -(ent.mean() if ent.dim() == 1 else ent.sum(dim=-1).mean())

        # L_policy-cloning = KL[μ ‖ π_θ] (replay only)
        if behavioral_cloning:
            if discrete:
                # Full KL using stored logits of μ
                # KL(μ‖π_θ) = Σ_a μ(a)[log μ(a) − log π_θ(a)]
                log_mu = replay_logits - torch.logsumexp(replay_logits, dim=-1, keepdim=True)
                log_pi = dist_s.logits    - torch.logsumexp(dist_s.logits,    dim=-1, keepdim=True)
                mu_probs = log_mu.exp()
                bc_policy_loss = (mu_probs * (log_mu - log_pi)).sum(dim=-1).mean()
            else:
                bc_policy_loss = F.mse_loss(log_prob, replay_log_probs_mu.detach())

            # L_value-cloning = ‖V_θ(s) − V_μ(s)‖² (replay only)
            # Hacky way of reconstructing historical value
            # V_μ(s_t) = advantage_t - v_s
            value_mu = replay_values_mu # replay_returns - replay_advantages  # replay_advantages - replay_returns
            bc_value_loss = F.mse_loss(value_s, value_mu)

        total_loss = (policy_gradient_loss
                    + vf_coef        * value_loss
                    + ent_coef       * entropy_loss)
        if behavioral_cloning:
            total_loss += bc_policy_coef * bc_policy_loss + bc_value_coef  * bc_value_loss

        return total_loss, bc_policy_loss if behavioral_cloning else None, bc_value_loss if behavioral_cloning else None, \
              policy_gradient_loss, value_loss, entropy_loss


    def learn(self) -> Dict[str, Any]:
        self.policy.train()
        if len(self.rollout_buffer) < self.rollout_buffer.capacity:
            return {}

        self.compute_returns_and_advantages()

        # E.g. given a ratio of 50% half of the batch_size is used on the on-policy
        # rollout and the other half on the replay data
        has_replay        = len(self.replay_buffer) > 0
        # new_batch_size    = max(1, round(self.new_replay_ratio * self.batch_size))
        # replay_batch_size = self.batch_size - new_batch_size if has_replay else 0

        info = {"value_loss": 0.0, "policy_loss": 0.0, "entropy": 0.0,
                "bc_policy_loss": 0.0, "bc_value_loss": 0.0}
        losses = []
        cnt = 0

        replay_rollout = self.replay_buffer.sample(policy=self.policy,
                                                   gamma=self.discount_factor,
                                                   gae_lambda=self.gae_lambda,
                                                   n_envs=self.n_envs,
                                                   batch_size=self.batch_size) if has_replay else None

        # fallback iterator: empty iterator if replay is None
        replay_iter = cycle(replay_rollout) if replay_rollout is not None else repeat((None, None, None, None, None, None, None))


        for _ in range(self.n_epochs):
            for (states, actions, old_log_probs, returns, advantages), \
                 (rp_states, rp_actions, rp_log_probs, rp_returns,
                  rp_advantages, rp_logits, rp_value_mu) in \
                    zip(self.rollout_buffer.get(self.batch_size), replay_iter):
                cnt += 1

                total_loss = None
                if self.use_ppo_loss:
                    # on-policy PPO surrogate loss
                    dist, values, _ = self.policy(states)
                    _actions = actions.squeeze(-1) if self.discrete else actions
                    lp = dist.log_prob(_actions)
                    log_probs    = (lp if lp.dim() == 1 else lp.sum(dim=-1)).unsqueeze(-1)
                    entropy_loss = (-dist.entropy().mean() if self.discrete
                                    else -dist.entropy().sum(dim=-1).mean())

                    ppo_loss, policy_loss, value_loss = self._compute_ppo_loss(
                        log_probs, old_log_probs, advantages, returns,
                        values, self.clip_range, self.vf_coef,
                        self.ent_coef, entropy_loss,
                    )
                    total_loss = ppo_loss
                else:
                    (loss, _, _, policy_loss, value_loss, entropy_loss) = self._compute_clear_loss(
                        states,
                        actions,
                        old_log_probs,
                        returns,
                        advantages,
                        None,

                        self.policy,
                        self.rho_bar,
                        self.vf_coef,
                        self.ent_coef,
                        self.bc_policy_coef,
                        self.bc_value_coef,
                        self.discrete,
                        behavioral_cloning=False
                    )
                    total_loss = loss

                bc_policy_loss = torch.tensor(0., device=self.device)
                bc_value_loss  = torch.tensor(0., device=self.device)
                if has_replay:
                    (replay_loss, bc_policy_loss, bc_value_loss, _, _, _) = self._compute_clear_loss(
                        rp_states,
                        rp_actions,
                        rp_log_probs,
                        rp_returns,
                        rp_advantages,
                        rp_logits,
                        rp_value_mu,
                        
                        self.policy,
                        self.rho_bar,
                        self.vf_coef,
                        self.ent_coef,
                        self.bc_policy_coef,
                        self.bc_value_coef,
                        self.discrete,
                        behavioral_cloning=self.behavioral_cloning
                    )
                    total_loss += replay_loss

                # gradient step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                losses.append(total_loss.item())
                info["value_loss"]     += value_loss.item()
                info["policy_loss"]    += policy_loss.item()
                info["entropy"]        += -entropy_loss.item()
                info["bc_policy_loss"] += bc_policy_loss.item() if bc_policy_loss is not None else 0.0
                info["bc_value_loss"]  += bc_value_loss.item() if bc_value_loss is not None else 0.0

        self.clear()

        info["loss"] = float(np.mean(losses))
        for k in ("value_loss", "policy_loss", "entropy", "bc_policy_loss", "bc_value_loss"):
            info[k] /= cnt
        return info
    
    def clear(self) -> None:
        dump_rollout_to_replay(self.rollout_buffer, self.replay_buffer)
        self.rollout_buffer.clear()

    def save(self, path: str) -> None:
        s = self.replay_buffer._size
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_buffer': {
                'obs':           self.replay_buffer.obs[:s].cpu(),
                'next_obs':      self.replay_buffer.next_obs[:s].cpu(),
                'actions':       self.replay_buffer.actions[:s].cpu(),
                'rewards':       self.replay_buffer.rewards[:s].cpu(),
                'dones':         self.replay_buffer.dones[:s].cpu(),
                'log_probs':     self.replay_buffer.log_probs[:s].cpu(),
                'values':        self.replay_buffer.values[:s].cpu(),
                'logits':        self.replay_buffer.logits[:s].cpu(),
                '_size':         s,
                '_total_seen':   self.replay_buffer._total_seen,
            },
            'hyperparameters': {
                'state_dimensions':       self.state_dimensions,
                'action_dimensions':      self.action_dimensions,
                'learning_rate':          self.learning_rate,
                'n_steps':                self.n_steps,
                'n_envs':                 self.n_envs,
                'n_epochs':               self.n_epochs,
                'gae_lambda':             self.gae_lambda,
                'clip_range':             self.clip_range,
                'ent_coef':               self.ent_coef,
                'vf_coef':                self.vf_coef,
                'max_grad_norm':          self.max_grad_norm,
                'replay_buffer_capacity': self.replay_buffer_capacity,
                'new_replay_ratio':       self.new_replay_ratio,
                'bc_policy_coef':         self.bc_policy_coef,
                'bc_value_coef':          self.bc_value_coef,
                'rho_bar':                self.rho_bar,
            }
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'replay_buffer' in checkpoint:
            rb = checkpoint['replay_buffer']
            s = rb['_size']
            self.replay_buffer.obs[:s]       = rb['obs'].to(self.device)
            self.replay_buffer.next_obs[:s]  = rb['next_obs'].to(self.device)
            self.replay_buffer.actions[:s]   = rb['actions'].to(self.device)
            self.replay_buffer.rewards[:s]   = rb['rewards'].to(self.device)
            self.replay_buffer.dones[:s]     = rb['dones'].to(self.device)
            self.replay_buffer.log_probs[:s] = rb['log_probs'].to(self.device)
            self.replay_buffer.values[:s]    = rb['values'].to(self.device)
            self.replay_buffer.logits[:s]    = rb['logits'].to(self.device)
            self.replay_buffer._size         = s
            self.replay_buffer._total_seen   = rb['_total_seen']
