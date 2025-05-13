import collections
import warnings
from functools import partial
from typing import Any, Optional, Tuple, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.policies import BasePolicy

from .utility import concat_numpy_action_for_step, concat_tensor_action_for_step


class HybridActorCriticPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Tuple,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space.spaces[1],
            # action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )

        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = dict(pi=[64, 64], vf=[64, 64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            # self.pi_features_extractor = self.features_extractor
            self.pi_features_extractor = self.features_extractor

            self.vf_features_extractor = self.features_extractor
        else:
            # self.pi_features_extractor = self.features_extractor
            self.pi_features_extractor = self.features_extractor

            self.vf_features_extractor = self.make_features_extractor()

        self.log_std_init = log_std_init
        dist_kwargs = None

        assert not (squash_output and not use_sde), "squash_output=True is only available when using gSDE (use_sde=True)"
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        # 检查动作空间是否为 Tuple[Discrete, Box], 依据离散与连续的部分生成网络
        assert isinstance(action_space, spaces.Tuple) and isinstance(action_space.spaces[0], spaces.Discrete) and isinstance(action_space.spaces[1], spaces.Box)
        # 离散分布
        self.action_dist_discrete = make_proba_distribution(action_space.spaces[0])
        # 连续分布
        self.action_dist_continue = make_proba_distribution(action_space.spaces[1], use_sde=use_sde, dist_kwargs=dist_kwargs)

        # self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

        self.hybrid_action_space = action_space
        self.is_hddpg_like_predict_mode = False

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)  # type: ignore[arg-type, return-value]

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(self.action_dist_continue, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist_continue.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # 连续部分
        # 高斯分布
        if isinstance(self.action_dist_continue, DiagGaussianDistribution):
            self.action_net_continue, self.log_std = self.action_dist_continue.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        # SGD 分布
        elif isinstance(self.action_dist_continue, StateDependentNoiseDistribution):
            self.action_net_continue, self.log_std = self.action_dist_continue.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        else:
            raise NotImplementedError(f"Unsupported continue distribution '{self.action_dist}'.")

        # 离散部分
        # 仅支持简单离散分布
        if isinstance(self.action_dist_discrete, (CategoricalDistribution)):
            self.action_net_discrete = self.action_dist_discrete.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported discrete distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),

                self.action_net_discrete: 0.01,
                self.action_net_continue: 0.01,

                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[tuple[th.Tensor, th.Tensor], th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        continue_distribution, discrete_distribution = self._get_action_dist_from_latent(latent_pi)

        # actions = distribution.get_actions(deterministic=deterministic)
        # log_prob = distribution.log_prob(actions)
        # actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        continue_actions = continue_distribution.get_actions(deterministic=deterministic)
        continue_log_prob = continue_distribution.log_prob(continue_actions)
        continue_actions = continue_actions.reshape((-1, *self.hybrid_action_space.spaces[1].shape))  # type: ignore[misc]

        discrete_actions = discrete_distribution.get_actions(deterministic=deterministic)
        discrete_log_prob = discrete_distribution.log_prob(discrete_actions)
        discrete_actions = discrete_actions.reshape((-1, *self.hybrid_action_space.spaces[0].shape))  # type: ignore[misc]

        return (continue_actions, discrete_actions), values, (continue_log_prob, discrete_log_prob)

    def extract_features(  # type: ignore[override]
        self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Union[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if self.share_features_extractor:
            return super().extract_features(obs, self.features_extractor if features_extractor is None else features_extractor)
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = super().extract_features(obs, self.pi_features_extractor)
            vf_features = super().extract_features(obs, self.vf_features_extractor)
            return pi_features, vf_features

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Tuple[Distribution, CategoricalDistribution]:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: (Continue Action distribution, Discrete Action distribution)
        """
        mean_actions_continue = self.action_net_continue(latent_pi)
        mean_actions_discrete = self.action_net_discrete(latent_pi)

        if isinstance(self.action_dist_continue, DiagGaussianDistribution):
            continue_action_dist = self.action_dist_continue.proba_distribution(mean_actions_continue, self.log_std)
        elif isinstance(self.action_dist_continue, StateDependentNoiseDistribution):
            continue_action_dist = self.action_dist_continue.proba_distribution(mean_actions_continue, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid continue action distribution")

        if isinstance(self.action_dist_discrete, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            discrete_action_dist = self.action_dist_discrete.proba_distribution(action_logits=mean_actions_discrete)
        else:
            raise ValueError("Invalid discrete action distribution")

        return (continue_action_dist, discrete_action_dist)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy, concat by discrete_action and continue_action
        """
        continue_distribution, discrete_distribution = self.get_distribution(observation)

        continue_action = continue_distribution.get_actions(deterministic)
        discrete_action = discrete_distribution.get_actions(deterministic)
    
        # print(continue_action.shape)
        return concat_tensor_action_for_step(discrete_action, continue_action)

    def evaluate_actions(self, obs: PyTorchObs, continue_actions: th.Tensor, discrete_actions: th.Tensor) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor], tuple[Optional[th.Tensor], Optional[th.Tensor]]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # distribution = self._get_action_dist_from_latent(latent_pi)
        # log_prob = distribution.log_prob(actions)
        # values = self.value_net(latent_vf)
        # entropy = distribution.entropy()

        values = self.value_net(latent_vf)
        continue_distribution, discrete_distribution = self._get_action_dist_from_latent(latent_pi)

        continue_log_prob = continue_distribution.log_prob(continue_actions)
        continue_entropy = continue_distribution.entropy()

        discrete_log_prob = discrete_distribution.log_prob(discrete_actions)
        discrete_entropy = discrete_distribution.entropy()

        return values, (continue_log_prob, discrete_log_prob), (continue_entropy, discrete_entropy)

    def get_distribution(self, obs: PyTorchObs) -> Tuple[Distribution, CategoricalDistribution]:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """

        if self.is_hddpg_like_predict_mode:
            return self.hpddpg_like_predict(observation, state, episode_start, deterministic)

        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(obs_tensor, deterministic=deterministic)
        # # Convert to numpy, and reshape to the original action shape
        # actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

        # if isinstance(self.action_space, spaces.Box):
        #     if self.squash_output:
        #         # Rescale to proper domain when using squashing
        #         actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
        #     else:
        #         # Actions could be on arbitrary scale, so clip the actions to avoid
        #         # out of bound error (e.g. if sampling from a Gaussian distribution)
        #         actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

        continue_actions = actions[:, 1:]
        discrete_actions = actions[:, 0]

        # Convert to numpy, and reshape to the original action shape
        continue_actions = continue_actions.cpu().numpy().reshape((-1, *self.hybrid_action_space.spaces[1].shape))  # type: ignore[misc, assignment]
        discrete_actions = discrete_actions.cpu().numpy().reshape((-1, *self.hybrid_action_space.spaces[0].shape))  # type: ignore[misc, assignment]

        # if isinstance(self.action_space, spaces.Box):
        if self.squash_output:
            # Rescale to proper domain when using squashing
            continue_actions = self.unscale_action(continue_actions)  # type: ignore[assignment, arg-type]
        else:
            # Actions could be on arbitrary scale, so clip the actions to avoid
            # out of bound error (e.g. if sampling from a Gaussian distribution)
            continue_actions = np.clip(continue_actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(continue_actions, np.ndarray)
            continue_actions = continue_actions.squeeze(axis=0)

            assert isinstance(discrete_actions, np.ndarray)
            discrete_actions = discrete_actions.squeeze(axis=0)
        
        return concat_numpy_action_for_step(discrete_actions, continue_actions), state

    def use_hddpg_like_predict_mode(self, is_use: bool = True):
        self.is_hddpg_like_predict_mode = is_use

    def hpddpg_like_predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            # actions = self._predict(obs_tensor)
            continue_distribution, discrete_distribution = self.get_distribution(obs_tensor)

            continue_actions = continue_distribution.get_actions(True)
            discrete_actions = discrete_distribution.distribution.probs
        
        # continue_actions = actions[:, 1:]
        # discrete_actions = actions[:, 0]

        # Convert to numpy, and reshape to the original action shape
        continue_actions = continue_actions.cpu().numpy().reshape((-1, *self.hybrid_action_space.spaces[1].shape))  # type: ignore[misc, assignment]
        discrete_actions = discrete_actions.cpu().numpy().reshape((-1, self.hybrid_action_space.spaces[0].n))  # type: ignore[misc, assignment]

        # if isinstance(self.action_space, spaces.Box):
        if self.squash_output:
            # Rescale to proper domain when using squashing
            continue_actions = self.unscale_action(continue_actions)  # type: ignore[assignment, arg-type]
        else:
            # Actions could be on arbitrary scale, so clip the actions to avoid
            # out of bound error (e.g. if sampling from a Gaussian distribution)
            continue_actions = np.clip(continue_actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(continue_actions, np.ndarray)
            continue_actions = continue_actions.squeeze(axis=0)

            assert isinstance(discrete_actions, np.ndarray)
            discrete_actions = discrete_actions.squeeze(axis=0)

            return np.concat((continue_actions, discrete_actions), axis = 1), state
        
        else:
            return np.concat((continue_actions, discrete_actions), axis = 1), state

        # print(continue_actions)
        # print(discrete_actions)

# class HybridActorCriticCnnPolicy(ActorCriticPolicy):
#     """
#     CNN policy class for actor-critic algorithms (has both policy and value prediction).
#     Used by A2C, PPO and the likes.

#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param lr_schedule: Learning rate schedule (could be constant)
#     :param net_arch: The specification of the policy and value networks.
#     :param activation_fn: Activation function
#     :param ortho_init: Whether to use or not orthogonal initialization
#     :param use_sde: Whether to use State Dependent Exploration or not
#     :param log_std_init: Initial value for the log standard deviation
#     :param full_std: Whether to use (n_features x n_actions) parameters
#         for the std instead of only (n_features,) when using gSDE
#     :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
#         a positive standard deviation (cf paper). It allows to keep variance
#         above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
#     :param squash_output: Whether to squash the output using a tanh function,
#         this allows to ensure boundaries when using gSDE.
#     :param features_extractor_class: Features extractor to use.
#     :param features_extractor_kwargs: Keyword arguments
#         to pass to the features extractor.
#     :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     """

#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Tuple,
#         lr_schedule: Schedule,
#         net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
#         activation_fn: type[nn.Module] = nn.Tanh,
#         ortho_init: bool = True,
#         use_sde: bool = False,
#         log_std_init: float = 0.0,
#         full_std: bool = True,
#         use_expln: bool = False,
#         squash_output: bool = False,
#         features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
#         features_extractor_kwargs: Optional[dict[str, Any]] = None,
#         share_features_extractor: bool = True,
#         normalize_images: bool = True,
#         optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[dict[str, Any]] = None,
#     ):
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             net_arch,
#             activation_fn,
#             ortho_init,
#             use_sde,
#             log_std_init,
#             full_std,
#             use_expln,
#             squash_output,
#             features_extractor_class,
#             features_extractor_kwargs,
#             share_features_extractor,
#             normalize_images,
#             optimizer_class,
#             optimizer_kwargs,
#         )
