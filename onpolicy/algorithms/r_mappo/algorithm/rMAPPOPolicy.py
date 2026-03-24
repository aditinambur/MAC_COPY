import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def _prepare_agent_messages(self, messages, obs_batch_size=None):
        """
        Convert shared agent messages to flattened per-agent aggregated messages.

        Expected shared shape: [n_envs, n_agents, n_agents, message_dim]
        Output shape: [n_envs * n_agents, message_dim]
        """
        if messages is None:
            return None

        if not torch.is_tensor(messages):
            messages = torch.as_tensor(messages, dtype=torch.float32, device=self.device)
        else:
            messages = messages.to(device=self.device, dtype=torch.float32)

        if messages.dim() == 4:
            n_envs, n_agents, _, message_dim = messages.shape
            eye = torch.eye(n_agents, device=messages.device, dtype=messages.dtype).view(1, n_agents, n_agents, 1)
            mask = 1.0 - eye
            denom = max(n_agents - 1, 1)
            # Mean over sender-axis while excluding self message.
            agent_messages = (messages * mask).sum(dim=2) / denom
            return agent_messages.reshape(n_envs * n_agents, message_dim)

        if messages.dim() == 3:
            bsz, n_agents, message_dim = messages.shape

            # Case A: flattened rollout/training batch [n_envs*n_agents, n_agents, message_dim].
            # Each row corresponds to a receiver agent; infer receiver index from row order and
            # aggregate messages from all other senders.
            if obs_batch_size is not None and bsz == obs_batch_size:
                if n_agents > 1:
                    recv_idx = torch.arange(bsz, device=messages.device) % n_agents
                    own_msg = messages[torch.arange(bsz, device=messages.device), recv_idx]
                    agent_messages = (messages.sum(dim=1) - own_msg) / (n_agents - 1)
                else:
                    agent_messages = torch.zeros(bsz, message_dim, device=messages.device, dtype=messages.dtype)
                return agent_messages

            # Case B: [n_envs, n_agents, message_dim], already per-agent tensors, flatten only.
            return messages.reshape(bsz * n_agents, message_dim)

        if messages.dim() == 2:
            # Already flattened [n_envs*n_agents, message_dim].
            return messages

        raise ValueError("Unexpected messages shape in policy: {}".format(tuple(messages.shape)))

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False, messages=None):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        :param messages: (np.ndarray) messages from other agents with shape [batch_size, n_agents, message_dim].
                                     If provided, will be used by actor for action generation (Phase 3 communication).

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        obs_batch_size = obs.shape[0] if hasattr(obs, "shape") else None
        agent_messages = self._prepare_agent_messages(messages, obs_batch_size=obs_batch_size)

        actor_out = self.actor(obs,
                               rnn_states_actor,
                               masks,
                               available_actions,
                               deterministic,
                       agent_messages)

        if len(actor_out) == 4:
            actions, action_log_probs, rnn_states_actor, output_messages = actor_out
        else:
            actions, action_log_probs, rnn_states_actor = actor_out
            output_messages = None

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        if output_messages is not None:
            return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, output_messages
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None, messages=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.
        :param messages: (torch.Tensor) messages from other agents with shape [batch_size, n_agents, message_dim].
                                       If provided, will be used by actor for action evaluation (Phase 3 communication).

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs_batch_size = obs.shape[0] if hasattr(obs, "shape") else None
        agent_messages = self._prepare_agent_messages(messages, obs_batch_size=obs_batch_size)

        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks,
                                         agent_messages)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actor_out = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        if len(actor_out) == 4:
            actions, _, rnn_states_actor, messages = actor_out
            return actions, rnn_states_actor, messages

        actions, _, rnn_states_actor = actor_out
        return actions, rnn_states_actor
