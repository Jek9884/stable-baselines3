from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import pandas as pd
import pickle
import numpy as np
import os
import cv2
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
import gymnasium as gym
import lightning as pl
from collections import deque
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy


SelfSAC = TypeVar("SelfSAC", bound="SAC")


class SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables


class SACMaster(SAC):
    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        policy_dir : str = None,
        experience_dir: str = None,
        descriptions_dir: str = None,
        ae_model: pl.LightningModule = None,
        task_description: str = None,
        tokenizer_str: str = None,
        similarity_thr: float = 0.5,
        k: int = None
    ):

        self.device = device
        self.tokenizer_str = tokenizer_str
        self.ae_model = ae_model
        self.task_description = task_description
        self.similarity_thr = similarity_thr
        self.k = k
        self.count = 0
        self.source_policy = None

        if k is not None:
            self.ae_model.to(self.device)

            self.frames_queue = deque(maxlen=4)

            if experience_dir is None or policy_dir is None or descriptions_dir is None:
                raise ValueError("Experience dir or Policy dir or Descriptions dir not defined")
            
            if ae_model is None:
                raise ValueError("Ae model not defined")

            if task_description is not None and tokenizer_str is None:
                raise ValueError("Tokenizer not defined")
            elif task_description is None and tokenizer_str is not None:
                raise ValueError("Task description not defined")
            elif task_description is not None and tokenizer_str is not None:
                self.task_description = self._get_description_cls(self.task_description)

            self.policies_pool = self._load_source_policies(policy_dir)
            self.count_source_policies = [0] * len(self.policies_pool)

            self.source_experience = self._load_source_experience(experience_dir, descriptions_dir)
        
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            ent_coef,
            target_update_interval,
            target_entropy,
            use_sde,
            sde_sample_freq,
            use_sde_at_warmup,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )
    
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"

            if self.k is not None:
                if self.count == self.k:
                    self.source_policy, policy_idx = self._select_best_source_policy()
                    self.count = 0

                    if self.source_policy is not None:
                        self.count_source_policies[policy_idx] += 1
                else:
                    self.count += 1

                if self.source_policy is None:
                    unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
                else:
                    obs = th.tensor(self._last_obs).to(self.device)
                    unscaled_action = self.policy(obs, deterministic=True)

                    unscaled_action = unscaled_action.cpu().detach().numpy()
            else:
                unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
    
    def _dump_logs(self) -> None:
        
        return super()._dump_logs()
    
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            if self.k is not None:
                # Get a frame from the environment
                frame = env.get_images()[-1]
                # Reshape frame
                np_frame = np.array(frame, dtype=np.uint8)
                np_frame = np_frame.reshape(3, 600, 600)
                # Cast from RGB to GRAYSCALE
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.frames_queue.append(frame)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _get_description_cls(self, description):

        tokenizer = BertTokenizer.from_pretrained(self.tokenizer_str)
        tokens = tokenizer(description, padding=True, truncation=True, return_tensors='pt', max_length=256)
        model = BertModel.from_pretrained(self.tokenizer_str)
        with th.no_grad():
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # Use [CLS] token embeddings
        cls_token = np.squeeze(embeddings)

        return cls_token

    def _get_env_name(self, name):

        if name == "intersection":
            return f'{name}-v1'
        else:
            if name == "lane":
                name = f'{name}-centering'
            return f'{name}-v0'
        
    def _apply_upsampling(self, image):
        
        # Dimensions of the original image
        original_height, original_width = image.shape

        # New dimensions
        new_height, new_width = 600, 600
        diff_height = new_height - original_height
        diff_width = new_width - original_width

        # Calculate the padding sizes for height and width
        top_pad = (new_height - original_height) // 2
        bottom_pad = new_height - original_height - top_pad
        left_pad = (new_width - original_width) // 2
        right_pad = new_width - original_width - left_pad
        
        background_color = 100

        # Create a new larger matrix filled with the padding value
        larger_matrix = np.full((new_height, new_width), background_color, dtype=image.dtype)
        
        # Copy the original image into the center of the larger matrix
        larger_matrix[top_pad:top_pad + original_height, left_pad:left_pad + original_width] = image

        return larger_matrix
    
    def _upsample_observations(self, x):

        vec_tmp = np.array(x)
        res = []
        if vec_tmp.shape[1] != 600 or vec_tmp.shape[2] != 600:
            for i in range(vec_tmp.shape[0]):
                res.append(self._apply_upsampling(vec_tmp[i]))
        else:
            res = x
        
        return res
    
    def _frame_to_embedding(self, x):
        
        images = np.array([x])
        images = th.from_numpy(images)
        images = images.to(th.float)

        images = images.to(th.device(self.device))
        self.ae_model.eval()

        # Make predictions
        with th.no_grad():
            res = self.ae_model(images, return_encodings=True)

        return np.squeeze(res.cpu().numpy())

    def _load_source_policies(self, policy_directory):

        KINEMATICS_OBSERVATION = {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy", "heading", "long_off", "lat_off", "ang_off"],
            "absolute": False,
            "order": "sorted",
        }

        config = {
            "observation": KINEMATICS_OBSERVATION,
            "action": {
                "type": "ContinuousAction",
            },
            "policy_frequency": 5, 
            "vehicles_count": 5,
            # "real_time_rendering":True,
        }

        policy_dict = {}
        files = os.listdir(policy_directory)
        for file in files:

            tokens = file.split('_')
            name = tokens[0]

            env_name = self._get_env_name(name)
            env = gym.make(env_name, config=config, render_mode="rgb_array")

            obs_space = env.observation_space

            if env_name == "lane-centering-v0":
                obs_space = spaces.Box(-np.inf, np.inf, (1,9), dtype = np.float32)

            scheduler = get_schedule_fn(0.1)
            policy = SACPolicy(obs_space, env.action_space, scheduler)
            policy.load_state_dict(th.load(f'{policy_directory}/{file}', map_location=th.device(self.device)))
            
            policy_dict[env_name] = policy

            env.close()

            print(f"{env_name} policy loaded!")

        return policy_dict

    def _load_source_experience(self, experience_directory, source_descriptions_dir, n_rows=10):

        files = os.listdir(experience_directory)

        df_descriptions = pd.read_json(source_descriptions_dir)

        df = None
        for file in files:

            with open(f'{experience_directory}/{file}', 'rb') as f:
                data = pickle.load(f)

            tokens = file.split('_')
            name = tokens[0]

            env_name = self._get_env_name(name)
            env_df = pd.DataFrame(data)

            # Get the environment description
            df_filtered = df_descriptions[df_descriptions['env_name'] == env_name]
            description = df_filtered['description'].iloc[0]
            if self.tokenizer_str is not None:
                description = self._get_description_cls(description)

            env_df['Description'] = [description] * len(env_df['Observation'])
            env_df['Env_name'] = [env_name] * len(env_df['Observation'])

            # Sort rows by the total reward and take the first n_rows
            env_df['Sum'] = env_df['Reward'].apply(np.sum)
            env_df = env_df.sort_values(by='Sum', ascending=False)
            env_df = env_df.head(n_rows)
            env_df = env_df.drop(columns=['Sum'])

            if df is None:
                df = env_df
            else:
                df = pd.concat([df, env_df], ignore_index=True)

        
        # Apply upsampling for the frames with a size smaller than 600x600
        df['Observation'] = df['Observation'].apply(self._upsample_observations)
        df['Observation'] = df['Observation'].apply(self._frame_to_embedding)
        print("Source experience loaded")

        return df

    def _apply_minmax(self, data, column):
        
        scaler = MinMaxScaler()
        data[column] = scaler.fit_transform(data[column].to_list()).tolist()

        return data
        
    def _compute_cosine(self, task_exp, source_exp):

        task_exp = np.array(task_exp)
        task_exp = task_exp.reshape(1, -1)

        source_exp = np.array(source_exp)

        res = cosine_similarity(source_exp, task_exp)

        return res.mean()
    
    def _concatenate_arrays(self, arr1, arr2):
        return np.concatenate((arr1, arr2))

    def _select_best_source_policy(self):

        # Reverses the list so that the frames are ordered from oldest to newest
        task_frames = list(reversed(self.frames_queue))
        # Cast frames into an embedding
        task_data = self._frame_to_embedding(task_frames)

        data = self.source_experience.copy(deep=True)
        data = data[['Observation', 'Description', 'Env_name']]

        # Add task experience to the experience dataframe
        new_row = {'Observation': task_data,
                    'Description': self.task_description,
                    'Env_name': 'target_task'}
        
        data.loc[len(data)] = new_row
        
        # Select the columns to use and apply MinMax scaling
        if self.task_description is not None:
            column = 'Concatenated'
            data[column] = data.apply(lambda row: self._concatenate_arrays(row['Observation'], row['Description']), axis=1)
            data = self._apply_minmax(data, column)
        else:
            column = 'Observation'
            data = self._apply_minmax(data, column)

        # Compute cosine between task data and each source data
        source_envs = list(self.policies_pool.keys())
        res = []
        for source_env in source_envs:
            env_data = data[data['Env_name'] == source_env][column]
            task_data = data[data['Env_name'] == 'target_task'][column]
            env_data = env_data.values.tolist()
            task_data = task_data.values.tolist()
            res.append(self._compute_cosine(task_data, env_data))
        
        # TODO if the maximum value is more than one, pick the max value index randomly
        # Find the max score
        max_value = max(res)

        # Returns a source policy if the score is over the threshold otherwise returns None
        if max_value >= self.similarity_thr:
            max_index = res.index(max_value)
            sim_env_name = source_envs[max_index]
            return self.policies_pool.get(sim_env_name), max_index
        else:
            return None