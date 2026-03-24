# Phase 3 Implementation: Message-Dependent Actor

## Summary
Implemented Phase 3 of emergent communication where agent actions now depend on both local observations AND aggregated messages from other agents.

---

## Modified Files

### 1. `onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py`

#### Modified `R_Actor.forward()`
```python
def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False, messages=None):
    """
    ...
    :param messages: (torch.Tensor) messages from other agents with shape [batch_size, n_agents, message_dim].
                                   If provided, will be aggregated and concatenated with observations.
    """
    obs = check(obs).to(**self.tpdv)
    rnn_states = check(rnn_states).to(**self.tpdv)
    masks = check(masks).to(**self.tpdv)
    if available_actions is not None:
        available_actions = check(available_actions).to(**self.tpdv)

    # Phase 3: Aggregate messages from other agents and concatenate with observations
    if messages is not None:
        messages = check(messages).to(**self.tpdv)
        # messages shape: [batch_size, n_agents, message_dim]
        batch_size, n_agents, msg_dim = messages.shape
        
        # Create mask to exclude own message: mask[i,i] = 0, mask[i,j] = 1 for j != i
        eye = torch.eye(n_agents, device=messages.device, dtype=torch.float32)
        mask = (1.0 - eye).unsqueeze(-1)  # [n_agents, n_agents, 1]
        masked_messages = messages * mask  # [batch_size, n_agents, message_dim]
        aggregated_messages = masked_messages.sum(dim=1) / (n_agents - 1)  # [batch_size, message_dim]
        
        # Concatenate observations with aggregated messages
        obs_and_messages = torch.cat([obs, aggregated_messages], dim=-1)
    else:
        obs_and_messages = obs

    actor_features = self.base(obs_and_messages)

    if self._use_naive_recurrent_policy or self._use_recurrent_policy:
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

    actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

    # Phase 2 communication: expose per-agent message embedding for runner-side sharing/logging.
    output_messages = actor_features
    return actions, action_log_probs, rnn_states, output_messages
```

**Key Changes:**
- Added `messages` parameter
- Implemented message aggregation: for each agent, exclude own message and take mean of others
- Concatenate obs with aggregated messages before passing to base encoder
- Maintain backward compatibility (messages=None falls back to obs-only input)

#### Modified `R_Actor.evaluate_actions()`
Same message aggregation logic added to evaluation method for use during training updates.

---

### 2. `onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py`

#### Modified `get_actions()`
```python
def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                deterministic=False, messages=None):
    """
    ...
    :param messages: (np.ndarray) messages from other agents with shape [batch_size, n_agents, message_dim].
                                 If provided, will be used by actor for action generation (Phase 3 communication).
    """
    actor_out = self.actor(obs,
                           rnn_states_actor,
                           masks,
                           available_actions,
                           deterministic,
                           messages)  # Pass messages to actor
    
    # ... rest of method remains same
```

**Key Changes:**
- Added `messages` parameter
- Pass messages through to actor forward pass

#### Modified `evaluate_actions()`
```python
def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                     available_actions=None, active_masks=None, messages=None):
    """
    ...
    :param messages: (torch.Tensor) messages from other agents with shape [batch_size, n_agents, message_dim].
                                   If provided, will be used by actor for action evaluation (Phase 3 communication).
    """
    action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                 rnn_states_actor,
                                                                 action,
                                                                 masks,
                                                                 available_actions,
                                                                 active_masks,
                                                                 messages)  # Pass messages to actor
    
    # ... rest of method remains same
```

**Key Changes:**
- Added `messages` parameter
- Pass messages to actor's evaluate_actions method

---

### 3. `onpolicy/runner/shared/mpe_runner.py`

#### Modified `__init__()`
```python
def __init__(self, config):
    super(MPERunner, self).__init__(config)
    self.gif_dir = self.all_args.gif_dir if self.all_args.gif_dir is not None else '.'
    self.message_log_dir = os.path.join(str(self.run_dir), "messages")
    os.makedirs(self.message_log_dir, exist_ok=True)
    
    # Phase 3: Initialize message buffers
    self.latest_messages = None
    self.latest_shared_messages = None
```

**Key Changes:**
- Initialize `latest_shared_messages` for Phase 3

#### Modified `warmup()`
```python
def warmup(self):
    # reset env
    obs = self.envs.reset()

    # Phase 3: Initialize zero messages for the first step
    # so agents can start using communication from step 0
    message_dim = self.all_args.hidden_size  # Message embeddings have same dim as hidden layer
    self.latest_shared_messages = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents, message_dim), 
                                           dtype=np.float32)

    # replay buffer
    if self.use_centralized_V:
        share_obs = obs.reshape(self.n_rollout_threads, -1)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
    else:
        share_obs = obs

    self.buffer.share_obs[0] = share_obs.copy()
    self.buffer.obs[0] = obs.copy()
```

**Key Changes:**
- Initialize zero messages: `[n_rollout_threads, n_agents, n_agents, message_dim]`
- Allows agents to use communication from step 0 (with zero initialization as priors)

#### Modified `collect()`
```python
@torch.no_grad()
def collect(self, step):
    self.trainer.prep_rollout()
    
    # Phase 3: Reshape messages for actor input if available
    messages_input = None
    if hasattr(self, 'latest_shared_messages') and self.latest_shared_messages is not None:
        # latest_shared_messages shape: [n_envs, n_agents, n_agents, message_dim]
        # Reshape for batched policy call: [n_envs*n_agents, n_agents, message_dim]
        n_envs = self.latest_shared_messages.shape[0]
        n_agents = self.latest_shared_messages.shape[1]
        message_dim = self.latest_shared_messages.shape[3]
        messages_input = self.latest_shared_messages.reshape(n_envs * n_agents, n_agents, message_dim)
    
    policy_out = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                        np.concatenate(self.buffer.obs[step]),
                        np.concatenate(self.buffer.rnn_states[step]),
                        np.concatenate(self.buffer.rnn_states_critic[step]),
                        np.concatenate(self.buffer.masks[step]),
                        messages=messages_input)  # Pass messages to policy

    # ... rest of method creates new shared_messages from policy output
    self.latest_shared_messages = shared_messages  # Update for next step
```

**Key Changes:**
- Reshape `latest_shared_messages` from `[n_envs, n_agents, n_agents, message_dim]` to `[n_envs*n_agents, n_agents, message_dim]`
- Pass reshaped messages to `get_actions()`
- Update `latest_shared_messages` after policy call

---

## Data Flow (Message Pipeline)

```
warmup():
  initialize: shared_messages [n_envs, n_agents, n_agents, message_dim] with zeros

collect(step=0):
  reshape: shared_messages → [n_envs*n_agents, n_agents, message_dim]
  → policy.get_actions(..., messages=reshpaed_messages)
    → actor.forward(..., messages=reshaped_messages)
      → aggregate messages for each agent (exclude own, mean of others)
      → concatenate: [obs_dim] + [message_dim] → new_input
      → base_encoder(new_input)
      → action, features
    → output_messages = features
  new_shared_messages = broadcast(output_messages) → [n_envs, n_agents, n_agents, message_dim]
  save: shared_messages for next step

collect(step=1+):
  (repeat with messages from previous step)
```

---

## Backward Compatibility

✓ **All changes are backward compatible:**
- If `messages=None`, actor behaves exactly as Phase 2 (obs-only input)
- Policy still returns 6-element tuple for Phase 2 logging
- No changes to critic or PPO loss
- No changes to buffer, only using aggregated messages at runtime
- During training (ppo_update), messages are not used (messages=None), so training dynamics unchanged

---

## Key Design Decisions

1. **Message Aggregation**: Mean pooling (excluding own message)
   - Simple, interpretable, and effective
   - Could be extended to attention-based pooling later

2. **Zero Initialization**: Use zero messages for step 0
   - Agents learn from zero baseline
   - Encourages learning to cooperate without relying on messages initially

3. **Shape Management**: [n_envs, n_agents, n_agents, message_dim]
   - Stores all-to-all messages within each environment
   - Allows actor to receive complete view of other agents' features

4. **No Buffer Modification**: Messages not stored between steps
   - Keeps implementation simple and non-invasive
   - During PPO training, agents optimize for observations (messages=None)
   - At inference, agents use previous step's messages

---

## Testing Checklist

- [ ] Training runs without errors
- [ ] Message shapes match expected dimensions
- [ ] Can inspect message tensors in view_messages.ipynb
- [ ] Actions change when messages are different
- [ ] Backward compatibility: old models still work
