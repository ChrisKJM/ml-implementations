import torch
from rl.policy_optimisation.base_policy_gradient import BasePolicyGradient


class VPG(BasePolicyGradient):
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
                 device: str | torch.device = "auto"):
        super().__init__(env, backbone, policy_head, value_head, optimiser, alpha=alpha, gamma=gamma, gae_lambda=gae_lambda, max_norm=max_norm, entropy_coeff=entropy_coeff, device=device)

    def calculate_psi(self, rewards, values, dones, advantages):
        mu = advantages.mean()
        sigma = advantages.std() + 1e-8
        return (advantages - mu) / sigma

    def calculate_policy_loss(self, log_probs, psi, old_log_probs=None):
        return -(log_probs * psi).mean()
