from gpytorch.mlls import DeepPredictiveLogLikelihood
from src.gp.acdeepsigma import ActorCriticDGP
from gpytorch.likelihoods import GaussianLikelihood
import torch

from src.gp.mll.deep_predictive_log_likelihood_rl import PolicyGradientDeepPredictiveLogLikelihood


class ActorCriticMLL:
    """
    Wrapper for RL training loop integration
    """

    def __init__(self, model: ActorCriticDGP, likelihood, num_data,
                 clip_range: float):
        self.mll_policy = PolicyGradientDeepPredictiveLogLikelihood(
            likelihood,
            model,
            num_data=num_data,
            clip_range=clip_range
        )

        # Can just use the regular likelihood for the value function.
        self.mll_value = DeepPredictiveLogLikelihood(
            likelihood,
            model,
            num_data=num_data
        )

        self.model = model

    def __call__(self, states, actions, advantages, returns, old_log_probs) -> torch.Tensor:
        """
        """
        policy_dist, value_dist = self.model(states)
        loss = -self.mll_policy(
            policy_dist,
            actions.squeeze(-1),
            adv=advantages.squeeze(-1),
            old_log_probs=old_log_probs
        ).mean() + -self.mll_value(
            value_dist,
            returns.squeeze(-1)
        ).mean()

        return loss
