from gpytorch.mlls import DeepPredictiveLogLikelihood


class DeepPredictiveLogLikelihoodRL(DeepPredictiveLogLikelihood):
    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        """
        Overwrite the log likelihood term to promote high return trajectories
        as motivated by the policy gradient theorem.
        """
        base_log_marginal = self.likelihood.log_marginal(target, approximate_dist_f, **kwargs)
        deep_log_marginal = self.model.quad_weights.unsqueeze(-1) + base_log_marginal

        deep_log_prob = deep_log_marginal.logsumexp(dim=0)
        # multiple deep_log_prob by return or an estimate of it.

        return deep_log_prob.sum(-1)