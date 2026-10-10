import torch
from rl.policy_optimisation.base_policy_gradient import BasePolicyGradient


class PPO(BasePolicyGradient):
    def __init__(self, env,
                 backbone,
                 policy_head,
                 value_head,
                 optimiser,
                 alpha: float = 1, # defines the scaling applied to the value network loss
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 max_norm: float = 0.5,
                 entropy_coeff: float = 0.01,
                 epsilon: float = 0.2,
                 device: str | torch.device = "auto"):
        super().__init__(env, backbone, policy_head, value_head, optimiser, alpha=alpha, gamma=gamma, gae_lambda=gae_lambda, max_norm=max_norm, entropy_coeff=entropy_coeff, allow_multiple_steps_per_epoch=True, device=device)
        self.epsilon = epsilon

    def calculate_psi(self, rewards, values, dones, advantages):
        mu = advantages.mean()
        sigma = advantages.std() + 1e-8
        return (advantages - mu) / sigma

    def calculate_policy_loss(self, log_probs, psi, old_log_probs=None):
        prob_ratio = torch.exp(log_probs - old_log_probs)
        return -(torch.minimum(prob_ratio * psi, torch.clamp(prob_ratio, 1 - self.epsilon, 1 + self.epsilon) * psi)).mean()
