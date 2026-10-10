import torch
from abc import ABC, abstractmethod


class BasePolicyGradient(ABC):
    def __init__(self, env,
                 backbone,
                 policy_head,
                 value_head,
                 optimiser,
                 alpha: float = 1, # defines the scaling applied to the value network loss
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 max_norm: float = 0,
                 entropy_coeff: float = 0,
                 allow_multiple_steps_per_epoch: bool = False,
                 device: str | torch.device = "auto"):
        self.env = env
        self.backbone = backbone
        self.policy_head = policy_head
        self.value_head = value_head
        self.optimiser = optimiser
        self.alpha = alpha
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_norm = max_norm
        self.entropy_coeff = entropy_coeff
        self.allow_multiple_steps_per_epoch = allow_multiple_steps_per_epoch

        if device == "auto":
            # try to infer the device based on model weights
            self.device = next(self.backbone.parameters()).device
        else:
            self.device = device

    @abstractmethod
    def calculate_psi(self, rewards, values, dones, advantages):
        ...

    @abstractmethod
    def calculate_policy_loss(self, log_probs, psi, old_log_probs=None):
        ...

    def calculate_value_loss(self, values, values_target):
        return torch.nn.functional.mse_loss(values, values_target)

    def _get_action_mask(self):
        if hasattr(self.env, "action_masks") and callable(self.env.action_masks):
            action_mask = self.env.action_masks()
            return torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).flatten()
        return None

    def _get_policy_distribution(self, backbone_out, action_mask):
        policy_out = self.policy_head(backbone_out)
        logits = policy_out.flatten(start_dim=1)

        if action_mask is not None:
            logits[action_mask == 0] = -1e8

        return torch.distributions.Categorical(logits=logits)

    def get_rollouts(self, rollout_length: int):
        if rollout_length <= 0:
            raise ValueError("Rollout length must be positive")

        observations = []
        action_masks = []
        actions = []
        dones = []
        rewards = []
        values = []
        log_probs = []

        obs, _ = self.env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        action_mask = self._get_action_mask()

        done = True
        for i in range(rollout_length):
            with torch.no_grad():
                backbone_out = self.backbone(obs[None, :])
                val = self.value_head(backbone_out).item() if self.value_head is not None else 0

                distr = self._get_policy_distribution(backbone_out, action_mask[None, :] if action_mask is not None else None)

                act = distr.sample()
                log_prob = distr.log_prob(act)

                observations.append(obs)
                action_masks.append(torch.ones(distr.logits.numel(), dtype=torch.bool, device=self.device) if action_mask is None else action_mask)
                actions.append(act.item())
                values.append(val)
                log_probs.append(log_prob.item())

                obs, rew, term, trunc, _ = self.env.step(act.item())
                done = term or trunc

                if done:
                    obs, _ = self.env.reset()

                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

                action_mask = self._get_action_mask()

                dones.append(done)
                rewards.append(rew)

        # handle finishing rollout collection on an unfinished rollout
        # calculates last_val used for bootstrapping in that case
        if done:
            last_val = 0
        elif self.value_head is not None:
            with torch.no_grad():
                last_val = self.value_head(self.backbone(obs[None, :])).item()
        else:
            last_val = 0

            discount = 1
            while not done:
                with torch.no_grad():
                    backbone_out = self.backbone(obs[None, :])

                    distr = self._get_policy_distribution(backbone_out, action_mask[None, :] if action_mask is not None else None)

                    act = distr.sample().item()

                    obs, rew, term, trunc, _ = self.env.step(act)
                    last_val += rew * discount
                    discount *= self.gamma
                    done = term or trunc

                    obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

                    action_mask = self._get_action_mask()

        observations = torch.stack(observations)
        action_masks = torch.stack(action_masks)
        actions = torch.tensor(actions, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)
        log_probs = torch.tensor(log_probs, dtype=torch.float32, device=self.device)

        return {
            "observations": observations,
            "action_masks": action_masks,
            "actions": actions,
            "dones": dones,
            "rewards": rewards,
            "values": values,
            "log_probs": log_probs,
            "last_val": last_val
        }

    def _calculate_gae(self, rewards, values, last_val, dones):
        advantages = torch.zeros_like(rewards, device=self.device)

        for i in range(len(rewards) - 1, -1, -1):
            not_done = 1 - dones[i]

            delta = rewards[i] + not_done * self.gamma * (last_val if i == len(rewards) - 1 else values[i + 1]) - values[i]
            advantages[i] = delta + not_done * self.gamma * self.gae_lambda * (advantages[i + 1] if i != len(rewards) - 1 else 0)

        return advantages

    def _calculate_rewards_to_go(self, rewards, dones):
        rewards_to_go = torch.zeros_like(rewards, device=self.device)

        for i in range(len(rewards) - 1, -1, -1):
            not_done = 1 - dones[i]

            rewards_to_go[i] = rewards[i] + not_done * self.gamma * (rewards_to_go[i + 1] if i != len(rewards) - 1 else 0)

        return rewards_to_go

    def train_epoch(self, batch_size: int, rollout_length: int, step_count: int = 1):
        if not self.allow_multiple_steps_per_epoch and step_count > 1:
            raise ValueError("This policy gradient algorithm does not support step_count > 1.")

        rollout_data = self.get_rollouts(rollout_length)

        observations = rollout_data["observations"]
        action_masks = rollout_data["action_masks"]
        actions = rollout_data["actions"]
        dones = rollout_data["dones"]
        rewards = rollout_data["rewards"]
        values_old = rollout_data["values"]
        log_probs_old = rollout_data["log_probs"]
        last_val = rollout_data["last_val"]

        # shouldn't be used in calculate_psi function if the model does not use the value head
        advantages = self._calculate_gae(rewards, values_old, last_val, dones)

        psi = self.calculate_psi(rewards, values_old, dones, advantages)

        values_target = advantages + values_old if self.value_head is not None else None

        for step in range(1, step_count + 1):
            shuffled_idxs = torch.randperm(rollout_length, device=self.device)
            for i in range(0, rollout_length, batch_size):
                batch_idxs = shuffled_idxs[i : i + batch_size]

                backbone_out = self.backbone(observations[batch_idxs])

                distr = self._get_policy_distribution(backbone_out, action_masks[batch_idxs])
                log_probs = distr.log_prob(actions[batch_idxs])
                policy_loss = self.calculate_policy_loss(log_probs, psi[batch_idxs], log_probs_old[batch_idxs])

                total_loss = policy_loss

                if self.entropy_coeff > 0:
                    entropy = distr.entropy().mean()
                    total_loss -= self.entropy_coeff * entropy

                if self.value_head is not None:
                    values = self.value_head(backbone_out).squeeze(-1)
                    value_loss = self.calculate_value_loss(values, values_target[batch_idxs])
                    total_loss += self.alpha * value_loss

                self.optimiser.zero_grad()
                total_loss.backward()
                if self.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.optimiser.param_groups[0]["params"], max_norm=self.max_norm)
                self.optimiser.step()
