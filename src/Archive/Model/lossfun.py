def loss_fn(self, parameters: torch.Tensor, v_batch: torch.Tensor, labels: torch.Tensor, missing_value = 0, use_boundary = False):
    mu = self.softplus_mu(parameters[:, :, 0])
    alpha = self.softplus_alpha(parameters[:, :, 1])
    zero_index = (labels != missing_value)
    distribution = torch.distributions.negative_binomial.NegativeBinomial(total_count=1/alpha[zero_index], logits=alpha[zero_index]*mu[zero_index])
    likelihood = distribution.log_prob(labels[zero_index])
    return -torch.mean(likelihood)



v = mu + (mu**2)/phi
phi * v = mu phi + (mu**2)
phi (v - mu) = mu^2
phi = mu^2/v-mu


v = mu + (mu^2) alpha





