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

        # Attention-based message aggregation: each agent learns which senders to
        # weight more heavily, instead of plain mean pooling.
        self.num_agents = args.num_agents
        message_dim = args.hidden_size
        self.attention_weight = torch.nn.Parameter(
            torch.randn(self.num_agents, message_dim, device=self.device) * 0.01
        )

        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + [self.attention_weight],
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

    def _apply_attention_aggregation(self, messages):
        """
        Aggregate sender-wise messages for each receiver using receiver-specific attention.

        :param messages: torch.Tensor shaped [batch_size, num_agents, message_dim], where
                         each row contains all senders' messages for one receiver.
        :return: torch.Tensor shaped [batch_size, message_dim]
        """
        bsz, n_agents, message_dim = messages.shape
        if n_agents <= 1:
            return torch.zeros(bsz, message_dim, device=messages.device, dtype=messages.dtype)

        recv_idx = torch.arange(bsz, device=messages.device) % n_agents
        recv_attention = self.attention_weight.to(device=messages.device, dtype=messages.dtype)[recv_idx]
        att_logits = torch.einsum('bsd,bd->bs', messages, recv_attention)

        self_mask = torch.eye(n_agents, device=messages.device, dtype=messages.dtype)[recv_idx]
        att_logits = att_logits.masked_fill(self_mask.bool(), float("-inf"))

        att_weights = torch.softmax(att_logits, dim=1)
        return torch.einsum('bs,bsd->bd', att_weights, messages)

    def _recompute_agent_messages_from_prev_share_obs(self, prev_share_obs):
        """
        Recompute sender messages from previous centralized observations so PPO gradients
        flow into the communication pathway during updates.
        """
        prev_share_obs = torch.as_tensor(prev_share_obs, dtype=torch.float32, device=self.device)
        prev_share_obs = torch.nan_to_num(prev_share_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        batch_size = prev_share_obs.shape[0]

        if self.num_agents <= 1 or self.actor.message_head is None:
            return torch.zeros(
                batch_size,
                self.actor.message_dim,
                device=self.device,
                dtype=prev_share_obs.dtype,
            )

        prev_obs_per_agent = prev_share_obs.view(batch_size, self.num_agents, self.actor.obs_dim)
        token_logits = self.actor.message_head(prev_obs_per_agent)
        token_probs = torch.softmax(token_logits, dim=-1)
        all_messages = torch.matmul(token_probs, self.actor.token_embedding.weight)
        recv_idx = torch.arange(batch_size, device=all_messages.device) % self.num_agents
        recv_attention = self.attention_weight.to(device=all_messages.device, dtype=all_messages.dtype)[recv_idx]
        att_logits = torch.einsum('bsd,bd->bs', all_messages, recv_attention)

        self_mask = torch.eye(self.num_agents, device=all_messages.device, dtype=all_messages.dtype)[recv_idx]
        att_logits = att_logits.masked_fill(self_mask.bool(), float("-inf"))

        att_weights = torch.softmax(att_logits, dim=1)
        return torch.einsum('bs,bsd->bd', att_weights, all_messages)

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
                return self._apply_attention_aggregation(messages)

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

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks, agent_messages)
        if output_messages is not None:
            return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, output_messages
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks, messages=None):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param messages: (torch.Tensor) messages from other agents with shape [batch_size, n_agents, message_dim].
                                       If provided, will be used by critic for value prediction.

        :return values: (torch.Tensor) value function predictions.
        """
        cent_obs_batch_size = cent_obs.shape[0] if hasattr(cent_obs, "shape") else None
        agent_messages = self._prepare_agent_messages(messages, obs_batch_size=cent_obs_batch_size)
        values, _ = self.critic(cent_obs, rnn_states_critic, masks, agent_messages)
        return values

    def get_action_distribution(self, obs, rnn_states_actor, masks, available_actions=None, messages=None):
        """
        Get the actor's action distribution(s) for the given inputs without sampling.
        Used for causal-influence analysis (e.g. KL divergence between distributions under
        different message interventions).
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param messages: (np.ndarray / torch.Tensor) messages from other agents.

        :return: a single distribution, or a list of distributions (multi_discrete / mixed action spaces).
        """
        obs_batch_size = obs.shape[0] if hasattr(obs, "shape") else None
        agent_messages = self._prepare_agent_messages(messages, obs_batch_size=obs_batch_size)
        return self.actor.get_action_distribution(obs, rnn_states_actor, masks, available_actions, agent_messages)

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None, messages=None, prev_share_obs=None):
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
        :param prev_share_obs: (torch.Tensor) centralized observation from the previous timestep, used to
                                       recompute messages through message_head so gradients flow to it.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs_batch_size = obs.shape[0] if hasattr(obs, "shape") else None
        if prev_share_obs is not None and self.actor.message_head is not None:
            agent_messages = self._recompute_agent_messages_from_prev_share_obs(prev_share_obs)
            prev_share_obs = None
        else:
            agent_messages = self._prepare_agent_messages(messages, obs_batch_size=obs_batch_size)

        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks,
                                         agent_messages,
                                         prev_share_obs)

        critic_messages = agent_messages.detach() if agent_messages is not None else None
        values, _ = self.critic(cent_obs, rnn_states_critic, masks, critic_messages)
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
