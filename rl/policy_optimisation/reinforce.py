import torch
from rl.policy_optimisation.base_policy_gradient import BasePolicyGradient


class REINFORCE(BasePolicyGradient):
    def __init__(self, env,
                 backbone,
                 policy_head,
                 optimiser,
                 gamma: float = 0.99,
                 device: str | torch.device = "auto"):
        super().__init__(env, backbone, policy_head, None, optimiser, gamma=gamma, device=device)

    def calculate_psi(self, rewards, values, dones, advantages):
        return self._calculate_rewards_to_go(rewards, dones)

    def calculate_policy_loss(self, log_probs, psi, old_log_probs=None):
        return -(log_probs * psi).mean()
