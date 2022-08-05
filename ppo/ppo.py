import os
import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm, HybridOnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy, HybridPolicy, NaivePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
            self,
            policy: Union[str, Union[Type[ActorCriticPolicy], Type[HybridPolicy], Type[NaivePolicy]]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):

        super(PPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        assert (
                batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                    buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        # DIY
        self.success_rate_threshold = 0.7

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(PPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))

        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "PPO",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            save_interval: Optional[int] = None,
            save_path: Optional[str] = None,
            save_count: int = 0,
    ) -> "OnPolicyAlgorithm":

        return super(PPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
            save_interval=save_interval,
            save_path=save_path,
            save_count=save_count,
        )


import time
from stable_baselines3.common.utils import safe_mean
from collections import deque


class HybridPPO(HybridOnPolicyAlgorithm):
    def __init__(
            self,
            policy: Union[str, Union[Type[ActorCriticPolicy], Type[HybridPolicy], Type[NaivePolicy]]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            is_two_stage_env: bool = False,
    ):

        super(HybridPPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
            is_two_stage_env=is_two_stage_env,
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        assert (
                batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                    buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        # DIY
        self.success_rate_threshold = 0.7
        self.removal_success_buffer = None
        self.global_success_buffer = None

        if _init_setup_model:
            self._setup_model()

        if is_two_stage_env:
            self.removal_success_buffer = deque(maxlen=100)
            self.global_success_buffer = deque(maxlen=100)

    def _setup_model(self) -> None:
        super(HybridPPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self, prefix=None):
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            if not continue_training:
                break
        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                           self.rollout_buffer.returns.flatten())
        # Logs
        prefix = f'{prefix}' if prefix is not None else ''

        self.logger.record(f"{prefix}train/entropy_loss", np.mean(entropy_losses))
        self.logger.record(f"{prefix}train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record(f"{prefix}train/value_loss", np.mean(value_losses))
        self.logger.record(f"{prefix}train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record(f"{prefix}train/clip_fraction", np.mean(clip_fractions))
        self.logger.record(f"{prefix}train/loss", loss.item())
        self.logger.record(f"{prefix}train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record(f"{prefix}train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record(f"{prefix}train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record(f"{prefix}train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record(f"{prefix}train/clip_range_vf", clip_range_vf)

    def train_estimate(self, prefix=None):
        assert self.is_two_stage_env

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        estimate_losses = []
        estimate_right_rates = []

        min_loss = th.inf
        loss_remain_times = 0
        buffer_size = self.env.num_envs * self.n_steps
        continue_training = True

        for _ in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                if loss_remain_times > buffer_size // self.batch_size:
                    continue_training = False
                    break
                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                success_rates_pred = self.policy.estimate_observations(rollout_data.observations).flatten()

                loss = F.binary_cross_entropy(success_rates_pred, rollout_data.is_successes)

                loss_item = loss.item()
                if loss_item < min_loss:
                    min_loss = loss_item
                    loss_remain_times = 0
                else:
                    loss_remain_times += 1
                estimate_losses.append(loss_item)


                is_successes_indicate = rollout_data.is_successes.long()
                cuda_success_rate_threshold = th.as_tensor(self.success_rate_threshold).to(self.device)
                pred_is_success_indicate = th.where(success_rates_pred <= cuda_success_rate_threshold,
                                                    success_rates_pred,
                                                    th.as_tensor(1, dtype=th.float).to(self.device)).to(self.device)
                pred_is_success_indicate = th.where(success_rates_pred > cuda_success_rate_threshold,
                                                    pred_is_success_indicate,
                                                    th.as_tensor(0, dtype=th.float).to(self.device)).to(self.device)

                estimate_right_rates.append((pred_is_success_indicate == is_successes_indicate)
                                            .float().mean().detach().cpu().numpy().item())

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.estimate_net.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            if not continue_training:
                break

        prefix = f'{prefix}' if prefix is not None else ''

        self.logger.record(f"{prefix}estimate/bce_loss", np.mean(estimate_losses))
        self.logger.record(f"{prefix}estimate/right_rate", np.mean(estimate_right_rates))

    def learn_one_step(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "OnPolicyAlgorithm",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            save_interval: Optional[int] = None,
            save_path: Optional[str] = None,
            accumulated_save_count: int = 0,
            accumulated_time_elapsed: float = 0.0,
            accumulated_iteration: int = 0,
            accumulated_total_timesteps: int = 0,
            prefix: str = None,
    ) -> "HybridPPO":
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        prefix = f'{prefix} ' if prefix is not None else ''

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            accumulated_iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and accumulated_iteration % log_interval == 0:
                fps = int(accumulated_total_timesteps + self.num_timesteps / (accumulated_time_elapsed + time.time() - self.start_time))
                self.logger.record(f"{prefix}time/iterations", accumulated_iteration, exclude="tensorboard")

                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record(f"{prefix}rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record(f"{prefix}rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    if len(self.ep_success_buffer) > 0:
                        self.logger.record(f"{prefix}rollout/success_rate", safe_mean(self.ep_success_buffer))

                    if self.is_two_stage_env:
                        if len(self.removal_success_buffer) > 0:
                            self.logger.record(f"{prefix}rollout/stage_1 success_rate", safe_mean(self.removal_success_buffer))
                        if len(self.global_success_buffer) > 0:
                            self.logger.record(f"{prefix}rollout/stage_2 success_rate", safe_mean(self.global_success_buffer))

                self.logger.record(f"{prefix}time/fps", fps)
                self.logger.record(f"{prefix}time/time_elapsed", int(accumulated_time_elapsed + time.time() - self.start_time),
                                   exclude="tensorboard")
                self.logger.record(f"{prefix}time/total_timesteps", accumulated_total_timesteps + self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=accumulated_total_timesteps + self.num_timesteps)

            # DIY
            if save_interval is not None and accumulated_iteration % save_interval == 0:
                assert save_path is not None
                accumulated_save_count += 1
                self.save(save_path + "_" + str(accumulated_save_count))
                self.logger.record(f"{prefix}Save Model", accumulated_save_count)
                self.logger.record(f"{prefix}time/iterations", accumulated_iteration)
                self.logger.record(f"{prefix}time/total_timesteps", accumulated_total_timesteps + self.num_timesteps)
                self.logger.dump(step=accumulated_total_timesteps + self.num_timesteps)

            self.train(prefix=prefix)

        callback.on_training_end()

        return self

    def setup_learn(self, total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name):
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )
        return total_timesteps, callback

    def learn_estimate(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "OnPolicyAlgorithm",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            save_interval: Optional[int] = None,
            save_path: Optional[str] = None,
            accumulated_save_count: int = 0,
            accumulated_time_elapsed: float = 0.0,
            accumulated_iteration: int = 0,
            accumulated_total_timesteps: int = 0,
            prefix: str = None,
    ) -> "HybridPPO":
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        prefix = f'{prefix} ' if prefix is not None else ''

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            accumulated_iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and accumulated_iteration % log_interval == 0:
                fps = int(accumulated_total_timesteps + self.num_timesteps / (accumulated_time_elapsed + time.time() - self.start_time))
                self.logger.record(f"{prefix}time/iterations", accumulated_iteration, exclude="tensorboard")

                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record(f"{prefix}rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record(f"{prefix}rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    if len(self.ep_success_buffer) > 0:
                        self.logger.record(f"{prefix}rollout/success_rate", safe_mean(self.ep_success_buffer))

                    if self.is_two_stage_env:
                        if len(self.removal_success_buffer) > 0:
                            self.logger.record(f"{prefix}rollout/stage_1 success_rate", safe_mean(self.removal_success_buffer))
                        if len(self.global_success_buffer) > 0:
                            self.logger.record(f"{prefix}rollout/stage_2 success_rate", safe_mean(self.global_success_buffer))

                self.logger.record(f"{prefix}time/fps", fps)
                self.logger.record(f"{prefix}time/time_elapsed", int(accumulated_time_elapsed + time.time() - self.start_time),
                                   exclude="tensorboard")
                self.logger.record(f"{prefix}time/total_timesteps", accumulated_total_timesteps + self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=accumulated_total_timesteps + self.num_timesteps)

            # DIY
            if save_interval is not None and accumulated_iteration % save_interval == 0:
                assert save_path is not None
                accumulated_save_count += 1
                self.save(save_path + "_" + str(accumulated_save_count))
                self.logger.record(f"{prefix}Save Model", accumulated_save_count)
                self.logger.record(f"{prefix}time/iterations", accumulated_iteration)
                self.logger.record(f"{prefix}time/total_timesteps", accumulated_total_timesteps + self.num_timesteps)
                self.logger.dump(step=accumulated_total_timesteps + self.num_timesteps)

            self.train_estimate(prefix=prefix)

        callback.on_training_end()

        return self

    def _two_stage_env_update_info_buffer(self, infos, dones):
        assert self.removal_success_buffer is not None and self.global_success_buffer is not None
        assert dones is not None

        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")

            maybe_removal_done = info.get('removal_done')
            maybe_removal_success = info.get('removal_success')

            maybe_global_done = info.get('global_done')
            maybe_global_success = info.get('global_success')

            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)
            if maybe_removal_success is not None and maybe_removal_done:
                self.removal_success_buffer.append(maybe_removal_success)
            if maybe_global_success is not None and maybe_global_done:
                self.global_success_buffer.append(maybe_global_success)


import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


lower = 0
upper = 1
indicate_list = [lower, upper]


def make_env(env_name, model_path=None):
    def _thunk():
        if model_path is not None:
            env = gym.make(env_name, model_path=model_path)
        else:
            env = gym.make(env_name)
        env = Monitor(env, None, allow_early_resets=True)

        return env

    return _thunk


def env_wrapper(env_name, num_envs, model_path=None):
    envs = [
        make_env(env_name, model_path)
        for _ in range(num_envs)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = VecNormalize(envs, norm_reward=False, norm_obs=False, training=False)

    return envs


class HrlPPO:
    def __init__(
            self,

            lower_policy: Union[str, Union[Type[ActorCriticPolicy], Type[HybridPolicy], Type[NaivePolicy]]],
            estimate_policy: Union[str, Union[Type[ActorCriticPolicy], Type[HybridPolicy], Type[NaivePolicy]]],
            upper_policy: Union[str, Union[Type[ActorCriticPolicy], Type[HybridPolicy], Type[NaivePolicy]]],

            lower_env: Union[GymEnv, str],
            estimate_env: Union[GymEnv, str],
            upper_env: Union[GymEnv, str],

            upper_env_id: str,
            upper_env_num: int = 3,
            upper_n_steps: int = 256,

            n_steps: int = 2048,
            learning_rate: Union[float, Schedule] = 3e-4,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        self.lower_agent = HybridPPO(lower_policy, lower_env, learning_rate, n_steps,
                                     batch_size, n_epochs, gamma, gae_lambda, clip_range, clip_range_vf, ent_coef,
                                     vf_coef, max_grad_norm, use_sde, sde_sample_freq, target_kl, tensorboard_log,
                                     create_eval_env, policy_kwargs, verbose, seed, device, _init_setup_model,
                                     is_two_stage_env=True)
        self.estimate_agent = HybridPPO(estimate_policy, estimate_env, learning_rate, n_steps,
                                        batch_size, n_epochs, gamma, gae_lambda, clip_range, clip_range_vf, ent_coef,
                                        vf_coef, max_grad_norm, use_sde, sde_sample_freq, target_kl, tensorboard_log,
                                        create_eval_env, policy_kwargs, verbose, seed, device, _init_setup_model,
                                        is_two_stage_env=True)
        self.upper_agent = HybridPPO(upper_policy, upper_env, learning_rate, n_steps,
                                     batch_size, n_epochs, gamma, gae_lambda, clip_range, clip_range_vf, ent_coef,
                                     vf_coef, max_grad_norm, use_sde, sde_sample_freq, target_kl, tensorboard_log,
                                     create_eval_env, policy_kwargs, verbose, seed, device, _init_setup_model,
                                     is_two_stage_env=False)

        self.wrapped_estimate_env = estimate_env
        self.upper_policy = upper_policy
        self.upper_env_id = upper_env_id
        self.upper_env_num = upper_env_num
        self.upper_n_steps = upper_n_steps
        self.upper_n_steps = upper_n_steps

        self.lr = learning_rate
        self.tensorboard_log = tensorboard_log

    def load_estimate(self, lower_model_path: str = None, logger=None):
        assert lower_model_path is not None and logger is not None, 'Model path and logger can not be None!!!'
        self.estimate_agent = HybridPPO.load(lower_model_path, env=self.wrapped_estimate_env, tensorboard_log=self.tensorboard_log)
        self.estimate_agent.set_logger(logger)

    def load_upper(self, lower_model_path: str = None, logger=None):
        assert lower_model_path is not None and logger is not None, 'Model path and logger can not be None!!!'
        wrapped_upper_env = env_wrapper(self.upper_env_id, self.upper_env_num, lower_model_path)
        self.upper_agent = HybridPPO(self.upper_policy, wrapped_upper_env, self.lr, self.upper_n_steps, verbose=1, tensorboard_log=self.tensorboard_log)
        self.upper_agent.set_logger(logger)

    def learn(
            self,
            total_iteration_count: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "PPO",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            lower_save_interval: Optional[int] = None,
            lower_save_path: Optional[str] = None,
            estimate_save_interval: Optional[int] = None,
            estimate_save_path: Optional[str] = None,
            upper_save_interval: Optional[int] = None,
            upper_save_path: Optional[str] = None,
            save_count: int = 0,
            train_lower_iteration: int = 100,
            train_estimate_iteration: int = 50,
            train_upper_iteration: int = 3000,
    ):
        lower_save_count = save_count
        lower_iteration = 0
        lower_total_timesteps = 0
        lower_time_elapsed = 0

        estimate_save_count = save_count
        estimate_iteration = 0
        estimate_total_timesteps = 0
        estimate_time_elapsed = 0

        upper_save_count = save_count
        upper_iteration = 0
        upper_total_timesteps = 0
        upper_time_elapsed = 0

        start_time = time.time()

        lower_single_steps = self.lower_agent.rollout_buffer.n_envs * self.lower_agent.rollout_buffer.buffer_size
        lower_total_steps = lower_single_steps * total_iteration_count
        lower_total_steps, lower_callback = self.lower_agent.setup_learn(total_timesteps=lower_total_steps,
                                                                         callback=callback,
                                                                         eval_env=eval_env,
                                                                         eval_freq=eval_freq,
                                                                         n_eval_episodes=n_eval_episodes,
                                                                         eval_log_path=eval_log_path,
                                                                         reset_num_timesteps=reset_num_timesteps,
                                                                         tb_log_name=f'Lower')

        estimate_single_steps = self.estimate_agent.rollout_buffer.n_envs * self.estimate_agent.rollout_buffer.buffer_size
        estimate_total_steps = estimate_single_steps * total_iteration_count
        estimate_total_steps, estimate_callback = self.estimate_agent.setup_learn(total_timesteps=estimate_total_steps,
                                                                         callback=callback,
                                                                         eval_env=eval_env, eval_freq=eval_freq,
                                                                         n_eval_episodes=n_eval_episodes,
                                                                         eval_log_path=eval_log_path,
                                                                         reset_num_timesteps=reset_num_timesteps,
                                                                         tb_log_name=f'Estimate')

        upper_single_steps = self.upper_agent.rollout_buffer.n_envs * self.upper_agent.rollout_buffer.buffer_size
        upper_total_steps = upper_single_steps * total_iteration_count
        upper_total_steps, upper_callback = self.upper_agent.setup_learn(total_timesteps=upper_total_steps,
                                                                         callback=callback,
                                                                         eval_env=eval_env, eval_freq=eval_freq,
                                                                         n_eval_episodes=n_eval_episodes,
                                                                         eval_log_path=eval_log_path,
                                                                         reset_num_timesteps=reset_num_timesteps,
                                                                         tb_log_name=f'Upper')

        estimate_callback.on_training_start(locals(), globals())
        lower_callback.on_training_start(locals(), globals())
        upper_callback.on_training_start(locals(), globals())

        for iteration in range(total_iteration_count):
            print(f'Round {iteration + 1} training starts!')

            self.lower_agent.set_logger(self.lower_agent.logger)
            self.lower_agent.learn_one_step(train_lower_iteration * lower_single_steps,
                                            lower_callback, log_interval, eval_env, eval_freq, n_eval_episodes, f'Lower', eval_log_path, reset_num_timesteps, lower_save_interval, lower_save_path,
                                            accumulated_save_count=lower_save_count,
                                            accumulated_time_elapsed=lower_time_elapsed,
                                            accumulated_iteration=lower_iteration,
                                            accumulated_total_timesteps=lower_total_timesteps,
                                            prefix='Lower')
            lower_save_count += train_lower_iteration // lower_save_interval
            lower_iteration += train_lower_iteration
            lower_time_elapsed += time.time() - self.lower_agent.start_time
            lower_total_steps += self.lower_agent.num_timesteps

            latest_lower_model_path = f'{lower_save_path}_{lower_save_count}'
            self.load_estimate(latest_lower_model_path, self.estimate_agent.logger)
            assert self.estimate_agent is not None
            estimate_single_steps = self.estimate_agent.rollout_buffer.n_envs * self.estimate_agent.rollout_buffer.buffer_size
            self.estimate_agent.learn_estimate(train_estimate_iteration * estimate_single_steps,
                                               estimate_callback, log_interval, eval_env, eval_freq, n_eval_episodes, f'Estimate', eval_log_path, reset_num_timesteps, estimate_save_interval, estimate_save_path,
                                               accumulated_save_count=estimate_save_count,
                                               accumulated_time_elapsed=estimate_time_elapsed,
                                               accumulated_iteration=estimate_iteration,
                                               accumulated_total_timesteps=estimate_total_timesteps,
                                               prefix='Estimate')
            estimate_save_count += train_estimate_iteration // estimate_save_interval
            estimate_iteration += train_estimate_iteration
            estimate_time_elapsed += time.time() - self.estimate_agent.start_time
            estimate_total_timesteps += self.estimate_agent.num_timesteps

            latest_estimate_model_path = f'{estimate_save_path}_{estimate_save_count}'
            self.load_upper(latest_estimate_model_path, self.upper_agent.logger)
            assert self.upper_agent is not None
            upper_single_steps = self.upper_agent.rollout_buffer.n_envs * self.upper_agent.rollout_buffer.buffer_size
            self.upper_agent.learn_one_step(train_upper_iteration * upper_single_steps,
                                            callback, log_interval, eval_env, eval_freq, n_eval_episodes, f'Upper', eval_log_path, reset_num_timesteps, upper_save_interval, upper_save_path,
                                            accumulated_save_count=upper_save_count,
                                            accumulated_time_elapsed=upper_time_elapsed,
                                            accumulated_iteration=upper_iteration,
                                            accumulated_total_timesteps=upper_total_timesteps,
                                            prefix='Upper')
            upper_save_count += train_upper_iteration // upper_save_interval
            upper_iteration += train_upper_iteration
            upper_time_elapsed += time.time() - self.upper_agent.start_time
            upper_total_timesteps += self.upper_agent.num_timesteps

            print(f'Round {iteration + 1} training ends!')
            print('-' * 64 + f' Total Time Elapsed: {time.time() - start_time} ' + '-' * 64)

        lower_callback.on_training_end()

    def load_agent(self, agent_name: str=None, model_path: str=None):
        assert agent_name is not None, 'Agent name can not be None!'
        assert model_path is not None, 'Model path can not be None!'
        agent_name_list = ['lower', 'estimate', 'upper']
        if agent_name == 'lower':
            self.lower_agent.load(model_path)
        elif agent_name == 'estimate':
            self.estimate_agent.load(model_path)
        elif agent_name == 'upper':
            self.upper_agent.load(model_path)
        else:
            assert False, f'Agent name is invalid!\nAgent name must in {agent_name_list}'

    def load_all_agent(self, model_dir: str=None):
        agent_name_list = ['lower', 'estimate', 'upper']
        agent_list = [self.lower_agent, self.estimate_agent, self.upper_agent]
        postfix = '.zip'
        for i in range(len(agent_name_list)):
            for model_path in os.listdir(model_dir):
                if agent_name_list[i] in model_path:
                    self.load_agent(agent_name_list[i], os.path.join(model_dir, model_path.replace(postfix, '')))
                    break
            assert isinstance(agent_list[i], HybridPPO), f'{agent_name_list[i]} agent load failed!'
