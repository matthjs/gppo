from gpytorch.mlls import DeepPredictiveLogLikelihood
from src.gp.acdeepsigma import ActorCriticDGP
from gpytorch.likelihoods import GaussianLikelihood
import torch


class PolicyGradientDeepPredictiveLogLikelihood(DeepPredictiveLogLikelihood):
    def __init__(self, likelihood, model, num_data, clip_range, beta):
        super().__init__(likelihood, model, num_data, beta)
        self.clip_range = clip_range
        self.last_log_prob = None

    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        """
        Modified log likelihood term for RL policy gradient updates.
        Incorporates advantage weighting for trajectory optimization.

        Args:
            approximate_dist_f: DSPP output distribution
            target: Action targets
            kwargs: Must contain 'adv' tensor of advantages

        Returns:
            Advantage-weighted log probabilities
        """
        # Get base log marginal from Gaussian likelihood
        base_log_marginal = self.likelihood.log_marginal(
            target,
            approximate_dist_f,
            **kwargs
        )

        # Combine with DSPP quadrature weights
        deep_log_marginal = self.model.quad_weights.unsqueeze(-1) + base_log_marginal

        # Compute deep log probabilities using logsumexp
        deep_log_prob = deep_log_marginal.logsumexp(dim=0)

        # Apply advantage weighting - key RL modification
        if 'adv' not in kwargs:
            raise ValueError("Must provide 'adv' tensor in kwargs for RL loss")
        if 'old_log_probs' not in kwargs:
            raise ValueError("Must provide 'old_log_probs' tensor in kwargs for RL loss")

        advantages = kwargs['adv']
        old_log_probs = kwargs['old_log_probs']
        if advantages.shape != deep_log_prob.shape:
            raise ValueError(f"Advantage shape {advantages.shape} must match "
                             f"log_prob shape {deep_log_prob.shape}")

        # ratio = torch.exp(deep_log_prob - old_log_probs)    # Note: Subtraction is division in log space.
        log_ratio = deep_log_prob - old_log_probs
        log_ratio_clamped = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio_clamped)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = torch.min(surr1, surr2).mean() #sum(-1)   # sum(-1) ?

        self.last_log_prob = deep_log_prob
        # Return advantage-weighted log probabilities
        return policy_loss



