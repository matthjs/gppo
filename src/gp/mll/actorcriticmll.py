from gpytorch.mlls import DeepPredictiveLogLikelihood
from src.gp.actorcriticdgp import ActorCriticDGP
import torch
from src.gp.mll.deep_predictive_log_likelihood_rl import PolicyGradientDeepPredictiveLogLikelihood


def conditional_squeeze(x: torch.Tensor) -> bool:
    if x.dim() > 1 and x.shape[1] == 1:
        return True

    return False


class ActorCriticMLL:
    """
    Composite loss wrapper for actor-critic training using a Deep Gaussian Process (DGP) model.
    Combines policy gradient loss, value prediction loss, and entropy bonus.

    :param model: Actor-critic model returning policy and value distributions.
    :param policy_likelihood: Likelihood function used for policy learning.
    :param value_likelihood: Likelihood function used for value function learning.
    :param num_data: Number of data points (=batch_size in online RL)
    :param clip_range: PPO clipping parameter.
    :param vf_coef: Weight for value loss term.
    :param ent_coef: Weight for entropy regularization.
    :param beta: Scales KL divergence term between prior and approximate posterior.
    """

    def __init__(self,
                 model: ActorCriticDGP,
                 policy_likelihood,
                 value_likelihood,
                 num_data,
                 clip_range: float, vf_coef: float, ent_coef: float,
                 beta: float):
        self.mll_policy = PolicyGradientDeepPredictiveLogLikelihood(
            policy_likelihood,
            model,
            num_data=num_data,
            clip_range=clip_range,
            beta=beta
        )

        # Can just use the regular likelihood for the value function.
        self.mll_value = DeepPredictiveLogLikelihood(
            value_likelihood,
            model,
            num_data=num_data,
            beta=beta
        )

        self.model = model
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    def __call__(self,
                 states: torch.Tensor,
                 actions: torch.Tensor,
                 advantages: torch.Tensor,
                 returns: torch.Tensor,
                 old_log_probs: torch.Tensor) -> tuple:
        """
        Compute total PPO-style loss.

        :param states: Batch of input states.
        :param actions: Taken actions.
        :param advantages: Estimated advantages.
        :param returns: Discounted returns (value targets).
        :param old_log_probs: Log-probabilities under old policy (for PPO).
        :return: Tuple of (total_loss, policy_loss, value_loss, entropy of policy distribution).
        """
        # conditionally squeeze()
        # DGP layers with a single output unit expect [batch_shape] targets while
        # D > 1 dimensional targets expect [batch_shape, D]
        do_squeeze = conditional_squeeze(actions)

        policy_dist, value_dist = self.model(states)
        # PPO clipped policy loss
        policy_loss = -self.mll_policy(
            policy_dist,
            actions.squeeze(-1) if do_squeeze else actions,
            adv=advantages,
            old_log_probs=old_log_probs
        ).mean()

        # Value function loss
        value_loss = -self.mll_value(
            value_dist,
            returns.squeeze(-1)
        ).mean()

        # entropy_loss is NEGATIVE entropy
        entropy_loss = -torch.mean(-self.mll_policy.last_log_prob)    # Need to use approximate entropy

        # Total loss: policy + value - entropy_bonus
        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

        return loss, policy_loss, value_loss, -entropy_loss
