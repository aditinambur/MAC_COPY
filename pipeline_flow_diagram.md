# MAC Project Pipeline and Data Flow

This document details the data flow and architectural pipeline of the Multi-Agent Emergent Communication (MAC) project. It contains two main sections: **Training and Rollout Flow** (where communication is learned) and the **Causal Influence of Communication (CIC) Evaluation Flow** (where message effects are measured).

---

## 1. Pipeline Overview Diagram

The diagram below shows how observations from the environment are converted into discrete message tokens, aggregated via attention, and used by the actor and critic. It also details the intervention-based causal evaluation step.

```mermaid
graph TD
    %% Training Loop
    subgraph Env [1. MPE Environment]
        State["Centralized State (s_t)"]
        Obs["Local Observations (o_t)"]
    end

    subgraph CommLayer [2. Learnable Communication Layer]
        MsgHead["message_head (Linear + Softmax)"]
        TokenEmb["token_embedding (Embedding Matrix)"]
        AttnAgg["Attention-based Aggregator"]
        
        Obs -->|Each agent i| MsgHead
        MsgHead -->|Vocabulary tokens| TokenEmb
        TokenEmb -->|Raw messages| AttnAgg
    end

    subgraph PolicyNetwork [3. Actor-Critic Networks]
        Actor["R_Actor (Policy)"]
        Critic["R_Critic (Centralized Value)"]
        
        Obs --> Actor
        State --> Critic
        AttnAgg -->|Aggregated incoming messages (m_t)| Actor
        AttnAgg -->|m_t| Critic
    end

    subgraph BufferTrain [4. Replay & Update]
        Buffer["Shared Buffer (stores obs, prev_share_obs, actions, messages)"]
        Trainer["R_MAPPO Trainer (PPO update)"]
        
        Actor -->|Actions & Values| Buffer
        Critic -->|Values| Buffer
        Buffer --> Trainer
        Trainer -->|Gradients backpropagate through Actor, Critic, & CommLayer| CommLayer
    end

    %% Causal Evaluation
    subgraph CIC [5. Causal Influence of Communication Evaluation]
        StateEval["Eval State (s)"]
        ObsEval["Eval Obs (o)"]
        RealMsg["Real Messages (m)"]
        ZeroMsg["Ablated Messages (0)"]
        
        DistReal["Real Distribution: P(a | o, m)"]
        DistZero["Ablated Distribution: Q(a | o, 0)"]
        
        ValReal["Real Value: V(s, m)"]
        ValZero["Ablated Value: V(s, 0)"]
        
        KL["Policy Sensitivity: KL( P || Q )"]
        VSens["Value Sensitivity: |V_real - V_zero|"]

        %% Connections
        ObsEval --> DistReal
        RealMsg --> DistReal
        ObsEval --> DistZero
        ZeroMsg --> DistZero
        
        StateEval --> ValReal
        RealMsg --> ValReal
        StateEval --> ValZero
        ZeroMsg --> ValZero
        
        DistReal & DistZero --> KL
        ValReal & ValZero --> VSens
        
        KL & VSens -->|Append row| CSV["causal_influence.csv"]
    end

    Env -->|Evaluation Mode| CIC
    Trainer -->|Update Weights| PolicyNetwork
```

---

## 2. Walkthrough of the Training Pipeline

During the training rollout step:

1.  **Observation Collection**: At timestep $t$, each agent receives its local observation $o_t$ from the MPE environment.
2.  **Message Generation**:
    *   The observation goes through `message_head` (a linear layer followed by `softmax` over the vocabulary size of 5 tokens).
    *   This selects a soft weighting over the shared `token_embedding` matrix (size `[vocab_size, message_dim]`).
    *   The raw message for agent $i$ is output.
3.  **Attention Aggregation**:
    *   For agent $i$, messages from all other agents $j \neq i$ are aggregated.
    *   Instead of standard average pooling, the model uses a learnable parameter `attention_weight` to weight the incoming messages.
    *   This produces the aggregated incoming message vector $m_{t, i}$.
4.  **Action & Value Output**:
    *   **Actor**: The local observation $o_t$ and the aggregated message $m_{t, i}$ are concatenated and forwarded to the policy network, outputting action probabilities.
    *   **Critic**: The centralized global state $s_t$ and aggregated messages $m_t$ are concatenated and passed to the centralized critic to estimate the state value $V_t$.
5.  **PPO Optimization & Gradient Coupled Communication**:
    *   To allow gradients to flow back into the communication layers, the PPO updates do not use cached messages from the rollout.
    *   Instead, in `R_Actor.evaluate_actions`, the messages are **recomputed online** from the stored previous observations.
    *   This ensures that the policy gradient updates both the actor and the `message_head` / `token_embedding` weights.

---

## 3. Walkthrough of the Causal Evaluation Pipeline

To determine how much the agents actually rely on the emergent communication channel, the [MPERunner._eval_causal_influence](file:///d:/capstone/capstone_project/MAC_COPY/onpolicy/runner/shared/mpe_runner.py#L385) method performs an intervention-based measurement:

1.  **Normal Trajectory Step**: The environment runs evaluation episodes using normal communication.
2.  **Counterfactual Query (The do-operator)**: At every step, for the *exact same state*, the policy is queried twice:
    *   **Real Path**: Evaluated with the real incoming messages $m \rightarrow P(a|o, m)$ and $V(s, m)$.
    *   **Ablated Path**: Evaluated with the communication channel zeroed out (simulating a complete breakdown/silence) $\rightarrow Q(a|o, 0)$ and $V(s, 0)$.
3.  **Causal Metric Calculation**:
    *   **KL Divergence**: $D_{KL}(P(a|o, m) \parallel Q(a|o, 0))$ is computed. A higher KL value indicates that the communication channel has a strong causal effect on the agent's actions.
    *   **Value Sensitivity**: $|V(s, m) - V(s, 0)|$ is computed. This measures how much the critic's state-value estimation depends on the messages.
4.  **Logging**: The means of these metrics across all agents are saved to `causal_influence.csv` at each evaluation interval.
