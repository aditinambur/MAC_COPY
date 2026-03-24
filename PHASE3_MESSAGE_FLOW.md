# Phase 3 Message Flow Visualization

## 1. Initialization (warmup)
```
┌─────────────────────────────────────┐
│ warmup():                           │
│  Initialize zero messages:          │
│  shape: [n_envs=1, n_agents=2,     │
│          n_agents=2, msg_dim=64]   │
│  values: all zeros                  │
└─────────────────────────────────────┘
           ↓
    [Step t=0]
```

## 2. Collect Step 0 (Initial Rollout)
```
┌────────────────────────────────────────────────────────────────┐
│ collect(step=0):                                               │
│                                                                │
│ 1. Reshape messages:                                           │
│    [1, 2, 2, 64] → [2, 2, 64]  (flatten envs)                 │
│                                                                │
│ 2. Call policy.get_actions(..., messages=[2,2,64])           │
│                                                                │
│    ┌────────────────────────────────────────────────────┐    │
│    │ Actor Forward Pass:                                │    │
│    │                                                    │    │
│    │ Input: obs=[2, obs_dim], messages=[2, 2, 64]      │    │
│    │                                                    │    │
│    │ For each agent i ∈ {0, 1}:                        │    │
│    │   - Get messages[i] = [agent0_msg, agent1_msg]    │    │
│    │   - Mask out messages[i,i] (own message)          │    │
│    │   - Aggregate: mean(messages[i, j≠i])             │    │
│    │   - Result: aggregated_msg[i] = [64]              │    │
│    │                                                    │    │
│    │ Aggregated shape: [2, 64]                          │    │
│    │                                                    │    │
│    │ 3. Concatenate:                                    │    │
│    │    input_to_encoder = concat([obs, agg_msgs])      │    │
│    │    shape: [2, obs_dim+64]                          │    │
│    │                                                    │    │
│    │ 4. Encoder processes concatenated input:           │    │
│    │    features = base_encoder(input_to_encoder)       │    │
│    │                                                    │    │
│    │ 5. Get actions and emit messages:                  │    │
│    │    actions = policy_head(features)                 │    │
│    │    output_messages = features = [2, 64]            │    │
│    └────────────────────────────────────────────────────┘    │
│                                                                │
│ 3. Reshape output for next step:                              │
│    output: [2, 64] (from agent 0,1)                          │
│    ↓                                                          │
│    split by env/agent: [1, 2, 64]                           │
│    ↓                                                          │
│    broadcast (repeat): [1, 2, 2, 64]                        │
│    (all-to-all within each env)                             │
│                                                                │
│ 4. Store: self.latest_shared_messages = [1, 2, 2, 64]       │
└────────────────────────────────────────────────────────────────┘
           ↓
    [Step t=1]
```

## 3. Collect Step 1 (Using Messages)
```
┌────────────────────────────────────────────────────────────────┐
│ collect(step=1):                                               │
│                                                                │
│ 1. Reshape messages from previous step:                        │
│    [1, 2, 2, 64] → [2, 2, 64]                               │
│    ↓                                                          │
│    Agent 0 sees: messages[0] = [agent0_features, agent1_feat] │
│    Agent 1 sees: messages[1] = [agent0_feat, agent1_features] │
│                                                                │
│ 2. Actor processes with real other-agent features:            │
│    For agent i:                                               │
│    ├─ Observation: current obs[i]                             │
│    ├─ Other message: aggregated_msg[i] from previous actor    │
│    └─ Action: depends on both local obs AND other agents      │
│                                                                │
│ 3. New output_messages (current actor features):              │
│    [2, 64] → broadcast → [1, 2, 2, 64]                       │
└────────────────────────────────────────────────────────────────┘
           ↓
    [Continue...]
```

## 4. Message Aggregation Detail (Per Agent)
```
Agent 0 Perspective:
    messages[0] = [
        [msg_dim_from_agent_0],  ← Own message (excluded)
        [msg_dim_from_agent_1]   ← Other agent (included)
    ]
    
    Aggregation:
    ├─ Create mask: [0, 1]  (exclude diagonal)
    ├─ Element-wise multiply: messages[0] * mask
    ├─ Sum: [msg_from_agent_1]
    ├─ Divide by (n_agents-1)=1: [msg_from_agent_1]
    └─ Result: aggregated[0] = [msg_from_agent_1]

Agent 1 Perspective:
    messages[1] = [
        [msg_from_agent_0]       ← Other agent (included)
        [msg_from_agent_1]       ← Own message (excluded)
    ]
    
    Result: aggregated[1] = [msg_from_agent_0]
```

## 5. Training Loop (No Change)
```
During ppo_update():
    evaluate_actions(..., messages=None)
    
Reason: messages not stored in buffer
        Would need to store messages at each step
        For now, keep it simple - train on observations only
        Messages influence actions at inference time
```

## 6. Shapes Summary
```
Warmup:
  latest_shared_messages = [n_envs, n_agents, n_agents, message_dim]

Collect:
  │
  ├─ Before policy call:
  │  messages_input = reshape([n_envs*n_agents, n_agents, message_dim])
  │
  ├─ Inside actor.forward():
  │  ├─ Input: messages [batch, n_agents, message_dim]
  │  ├─ Per agent: exclude own, aggregate others
  │  ├─ aggregated_messages [batch, message_dim]
  │  ├─ Concatenate: [batch, obs_dim+message_dim]
  │  └─ Output: actions, messages
  │
  ├─ After policy call:
  │  output_messages [batch, message_dim]
  │  ↓ reshape and broadcast
  │  latest_shared_messages [n_envs, n_agents, n_agents, message_dim]
  │
  └─ Next iteration uses latest_shared_messages
```

## Key Properties

**Causal Structure:**
- Messages at step t come from actor outputs at step t-1
- Agents see what other agents "thought" at previous step
- Allows temporal coordination

**Information Flow:**
- Agent 0 receives aggregated features from Agent 1 (from prev step)
- Agent 1 receives aggregated features from Agent 0 (from prev step)
- Enables emergent communication without explicit message format

**Learning Dynamics:**
- Actor learns to produce meaningful features in messages
- Actor also learns to use messages to improve actions
- Both happen simultaneously through backprop
- No explicit communication loss (yet) - purely through action improvement

**Initialization:**
- Zero messages ensure agents learn to cooperate from scratch
- Prevents training from failing due to random communication noise
- Encourages baseline policies that work with zero signals
