import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        print(obs_shape)
        base = CNNBase if len(obs_shape) == 3 else MLPBase

        self._supports_message_concat = (base is MLPBase and len(obs_shape) == 1)
        self.message_dim = self.hidden_size
        self.obs_dim = int(obs_shape[0]) if len(obs_shape) == 1 else None

        # Discrete token-based communication: agents emit a distribution over a shared
        # vocabulary, which is embedded into message_dim-sized vectors. This forces all
        # agents to communicate using the same set of "words" (token embeddings).
        self.vocab_size = 5
        self.message_head = nn.Linear(self.obs_dim, self.vocab_size) if self.obs_dim is not None else None
        self.token_embedding = nn.Embedding(self.vocab_size, self.message_dim) if self.obs_dim is not None else None

        self.num_agents = args.num_agents

        if self._supports_message_concat:
            # Build encoder for [obs, message] concatenated input.
            fused_obs_shape = (self.obs_dim + self.message_dim,)
            self.base = base(args, fused_obs_shape)
        else:
            self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False, messages=None):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.
        :param messages: (torch.Tensor) messages from other agents with shape [batch_size, n_agents, message_dim].
                           If provided, will be aggregated and used to modulate actor features.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        :return messages: (torch.Tensor) message embeddings for Phase 2 communication.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        # Phase 3: Build encoder input as concat(obs, aggregated_messages).
        if self._supports_message_concat:
            if messages is None:
                messages = torch.zeros(obs.shape[0], self.message_dim, device=obs.device, dtype=obs.dtype)
            else:
                messages = check(messages).to(**self.tpdv)
                messages = torch.nan_to_num(messages, nan=0.0, posinf=1.0, neginf=-1.0)

            actor_input = torch.cat([obs, messages], dim=-1)
            actor_features = self.base(actor_input)
        else:
            # Fallback path for non-MLP encoders.
            actor_features = self.base(obs)

        actor_features = torch.nan_to_num(actor_features, nan=0.0, posinf=1.0, neginf=-1.0)
        if self.message_head is not None:
            # Emit discrete token-based messages from local observations only:
            # message = softmax(message_head(obs)) @ token_embedding.weight
            token_logits = self.message_head(obs)  # [batch, vocab_size]
            token_probs = torch.softmax(token_logits, dim=-1)  # [batch, vocab_size]
            output_messages = torch.matmul(token_probs, self.token_embedding.weight)  # [batch, message_dim]
        else:
            output_messages = torch.tanh(actor_features)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        # Phase 2 communication: expose per-agent message embedding for runner-side sharing/logging.
        return actions, action_log_probs, rnn_states, output_messages

    def get_action_distribution(self, obs, rnn_states, masks, available_actions=None, messages=None):
        """
        Compute the action distribution(s) for the given inputs without sampling, for use in
        causal-influence analyses (e.g. KL divergence between distributions under different
        message interventions).
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param messages: (torch.Tensor) aggregated messages, shape [batch_size, message_dim].

        :return: a single distribution, or a list of distributions (multi_discrete / mixed action spaces).
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self._supports_message_concat:
            if messages is None:
                messages = torch.zeros(obs.shape[0], self.message_dim, device=obs.device, dtype=obs.dtype)
            else:
                messages = check(messages).to(**self.tpdv)
                messages = torch.nan_to_num(messages, nan=0.0, posinf=1.0, neginf=-1.0)

            actor_input = torch.cat([obs, messages], dim=-1)
            actor_features = self.base(actor_input)
        else:
            actor_features = self.base(obs)

        actor_features = torch.nan_to_num(actor_features, nan=0.0, posinf=1.0, neginf=-1.0)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        return self.act.get_distributions(actor_features, available_actions)

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None,
                         messages=None, prev_share_obs=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.
        :param messages: (torch.Tensor) messages from other agents with shape [batch_size, n_agents, message_dim].
                           If provided, will be aggregated and used to modulate actor features.
        :param prev_share_obs: (torch.Tensor) centralized observation from the previous timestep, shape
                           [batch_size, num_agents * obs_dim]. If provided (and message_head exists), the
                           aggregated message is recomputed from these observations through message_head so
                           that PPO gradients flow into message_head.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # Phase 3: Build encoder input as concat(obs, aggregated_messages).
        if self._supports_message_concat:
            if prev_share_obs is not None and self.message_head is not None:
                # Recompute messages from previous-timestep observations through message_head
                # so that PPO gradients flow back into message_head's parameters.
                prev_share_obs = check(prev_share_obs).to(**self.tpdv)
                prev_share_obs = torch.nan_to_num(prev_share_obs, nan=0.0, posinf=1.0, neginf=-1.0)
                batch_size = prev_share_obs.shape[0]

                if self.num_agents > 1:
                    prev_obs_per_agent = prev_share_obs.view(batch_size, self.num_agents, self.obs_dim)

                    token_logits = self.message_head(prev_obs_per_agent)  # [batch, num_agents, vocab_size]
                    token_probs = torch.softmax(token_logits, dim=-1)
                    all_messages = torch.matmul(token_probs, self.token_embedding.weight)  # [batch, num_agents, message_dim]

                    # Aggregate excluding self-messages
                    recv_idx = torch.arange(batch_size, device=obs.device) % self.num_agents
                    eye = torch.eye(self.num_agents, device=obs.device, dtype=all_messages.dtype)[recv_idx]
                    other_mask = (1.0 - eye).unsqueeze(-1)
                    messages = (all_messages * other_mask).sum(dim=1) / (self.num_agents - 1)
                else:
                    messages = torch.zeros(batch_size, self.message_dim, device=obs.device, dtype=obs.dtype)
            elif messages is None:
                messages = torch.zeros(obs.shape[0], self.message_dim, device=obs.device, dtype=obs.dtype)
            else:
                messages = check(messages).to(**self.tpdv)
                messages = torch.nan_to_num(messages, nan=0.0, posinf=1.0, neginf=-1.0)

            actor_input = torch.cat([obs, messages], dim=-1)
            actor_features = self.base(actor_input)
        else:
            # Fallback path for non-MLP encoders.
            actor_features = self.base(obs)

        actor_features = torch.nan_to_num(actor_features, nan=0.0, posinf=1.0, neginf=-1.0)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        print(cent_obs_shape)

        # Message-aware critic: expand input to include aggregated messages so that the
        # value function can credit (or penalize) communication.
        self._supports_message_concat = (base is MLPBase and len(cent_obs_shape) == 1)
        self.message_dim = self.hidden_size

        if self._supports_message_concat:
            cent_obs_shape = (cent_obs_shape[0] + self.message_dim,)

        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks, messages=None):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.
        :param messages: (torch.Tensor) aggregated messages from other agents, shape [batch, message_dim].

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self._supports_message_concat:
            if messages is None:
                messages = torch.zeros(cent_obs.shape[0], self.message_dim, device=cent_obs.device, dtype=cent_obs.dtype)
            else:
                messages = check(messages).to(**self.tpdv)
                messages = torch.nan_to_num(messages, nan=0.0, posinf=1.0, neginf=-1.0)

            critic_input = torch.cat([cent_obs, messages], dim=-1)
        else:
            critic_input = cent_obs

        critic_features = self.base(critic_input)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states
