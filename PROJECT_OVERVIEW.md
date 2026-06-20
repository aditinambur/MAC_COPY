# Project Overview — Causal Mechanistic Emergent Language (MAC_COPY)

This is the single, authoritative document for the capstone project as it currently
stands in **this repository (`MAC_COPY`)**. It supersedes the older scattered notes
(`COMMUNICATION_CHANGES.md`, `PHASE3_COMPLETE_DOCUMENTATION.md`,
`CAUSAL_EVALUATION_COMPLETE.md`, `EXPERIMENTS.md`). Only the parts that are actually
implemented and verified in this repo are recorded here.

---

## 1. Project Aim

**Causal Mechanistic Emergent Language with Online Repair and Meta-Causal Adaptation
in Multi-Agent Reinforcement Learning.**

Build a MARL communication system that:

1. **Learns communication** — agents emit and consume messages that improve cooperation.
2. **Measures the causal effect of messages** — quantify how much each agent's messages
   actually change other agents' behavior and value estimates (not just reward).
3. **Repairs degraded communication** under environment changes *(future work)*.
4. **Adapts online** via meta-causal adaptation *(future work)*.

The emphasis is on **causal communication + repair**, not merely maximizing reward.

**Scenario / setup this project targets:**

- Environment: **MPE `simple_spread`** (cooperative navigation)
- Algorithm: **`mappo`** (MAPPO, recurrent actor-critic)
- Entry point: **`onpolicy/scripts/train/train_mpe.py`**
- Agents: **2** (default `--num_agents 2`)
- Observation dim: 14 per agent; centralized obs dim: 28; message dim = `hidden_size` = 64;
  vocabulary size: 5 tokens.

---

## 2. Roadmap & Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Communication architecture** | Learnable, gradient-trained, discrete-token communication with attention aggregation and a message-aware critic | ✅ Implemented & verified |
| **Phase 1 — Causal Influence of Communication (CIC)** | Intervention-based per-agent measurement of message effect on policy (KL) and value (ΔV) | ✅ Implemented & verified |
| **Phase 2 — Degradation detection** | Trigger when causal influence drops vs. a rolling baseline | ⏳ Planned |
| **Phase 3 — Repair mechanism** | Conditional repair (entropy re-anneal / partial re-init / vocab expansion) when degradation detected | ⏳ Planned |
| **Phase 4 — Meta-causal adaptation** | Meta-train the repair controller across a distribution of perturbations | ⏳ Planned |

---

## 3. Files to Look At (this project only)

The project runs through the **`mappo` → `r_mappo`** algorithm stack. These are the
files that matter:

### Entry point & configuration
- **`onpolicy/scripts/train/train_mpe.py`** — training/eval entry point; builds envs,
  runner, and `run_dir` (where results are stored).
- **`onpolicy/config.py`** — all command-line flags, including the communication and
  causal-analysis flags (see §6).

### Communication architecture (the "causal mechanistic" message layer)
- **`onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py`**
  - `R_Actor` — discrete-token `message_head` + `token_embedding`, message-conditioned
    actor, gradient-coupled message recomputation in `evaluate_actions`,
    and `get_action_distribution` (used by CIC).
  - `R_Critic` — message-aware critic (centralized obs concatenated with aggregated
    messages).
- **`onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py`** — `R_MAPPOPolicy`:
  attention-based message aggregation (`_prepare_agent_messages`), the learnable
  `attention_weight` parameter, and message threading through `get_actions`,
  `get_values`, `evaluate_actions`, `get_action_distribution`.
- **`onpolicy/algorithms/r_mappo/r_mappo.py`** — `R_MAPPO` trainer; `ppo_update`
  unpacks `prev_share_obs_batch` and forwards it for gradient flow into `message_head`.
- **`onpolicy/algorithms/utils/act.py`** — `ACTLayer.get_distributions()` (raw action
  distribution for KL computation).
- **`onpolicy/utils/shared_buffer.py`** — stores `aggregated_messages` and provides
  `prev_share_obs` from `feed_forward_generator()`.

### Runner & measurement
- **`onpolicy/runner/shared/mpe_runner.py`** — rollout (`collect`), message aggregation,
  evaluation (`eval`), the **causal-influence measurement (`_eval_causal_influence`)**,
  and CSV logging (`_save_causal_influence_csv`).
- **`onpolicy/runner/shared/base_runner.py`** — `run_dir` setup and `log_env` /
  `log_train` (wandb / TensorBoard logging).

---

## 4. What Is Implemented

### 4.1 Communication architecture

A learnable, gradient-trained, discrete-token communication system with attention-based
aggregation and a message-aware critic.

1. **Discrete token-based communication (shared vocabulary)** — `R_Actor`
   - `vocab_size = 5`, `message_head = nn.Linear(obs_dim, vocab_size)`,
     `token_embedding = nn.Embedding(vocab_size, message_dim)`.
   - A message is `softmax(message_head(obs)) @ token_embedding.weight` — a weighted
     combination of shared "word" embeddings. All agents draw from the same vocabulary.

2. **Attention-based message aggregation** — `R_MAPPOPolicy`
   - Learnable `attention_weight` parameter of shape `[num_agents, message_dim]`,
     trained alongside the actor.
   - Each receiver agent learns which senders to weight, instead of plain mean pooling
     (self-messages are masked out).

3. **Gradient flow into `message_head` during PPO** — `R_Actor.evaluate_actions` +
   `SharedReplayBuffer`
   - The buffer yields `prev_share_obs` (previous-timestep centralized obs).
   - During the PPO update, messages are **recomputed** from `prev_share_obs` through
     `message_head`/`token_embedding`, so policy-loss gradients reach the communication
     parameters (the stored numpy messages carry no gradient).

4. **Message-aware critic** — `R_Critic`
   - When using an MLP critic with flat centralized obs, the input is expanded by
     `message_dim`; aggregated messages are concatenated onto `cent_obs` before the
     value MLP, so the value function can credit/penalize communication.

5. **Trainer wiring** — `R_MAPPO.ppo_update`
   - `feed_forward_generator` yields 14 elements (adds `prev_share_obs_batch`);
     `ppo_update` forwards it to `policy.evaluate_actions(..., prev_share_obs=...)`.

### 4.2 Phase 1 — Causal Influence of Communication (CIC)

This is the **correct, verified causal-influence measurement** for the project. It is an
*intervention-based* measurement; it adds **no new network layer**. During evaluation,
the trajectory follows the normal (non-intervened) policy, but at every step the trained
policy is queried on the **same state twice**:

- once with the **real** incoming aggregated message, and
- once with that message **ablated (zeroed)** — the do-operator on the message channel.

Two per-agent quantities are recorded:

- **`causal_influence_kl_agent{i}`** — KL divergence between the agent's action
  distribution with vs. without its incoming message. Measures how much the **policy**
  causally depends on communication.
- **`causal_influence_value_sensitivity_agent{i}`** — `|V_real − V_ablated|`. Measures
  how much the **critic's value estimate** causally depends on communication.

Plus the across-agent means `causal_influence_kl_mean` and
`causal_influence_value_sensitivity_mean`.

**Implementation:** `MPERunner._eval_causal_influence()` in
`onpolicy/runner/shared/mpe_runner.py`, supported by
`R_MAPPOPolicy.get_action_distribution()`, `R_Actor.get_action_distribution()`, and
`ACTLayer.get_distributions()`. It is run automatically inside `eval()` (on by default).

**Verified behavior:** causal influence grows with training — e.g. in a short run, mean
KL rose from ~0.004 → ~0.048 and mean value-sensitivity from ~0.24 → ~0.61 between the
first and second eval — exactly the trend Phase 2 will threshold on.

> **Note on the reward-level ablations.** The `--eval_disable_messages` and
> `--eval_noise_std` flags also run reward-only eval passes
> (`eval_normal_rewards` / `eval_no_message_rewards` / `eval_noisy_message_rewards`).
> These are **coarse reward-level robustness checks**, not the project's causal-influence
> measure. The rigorous, per-agent causal signal is the CIC (KL + value sensitivity)
> described above.

---

## 5. How to Run

### Prerequisites (one-time, Windows + conda)

1. **Editable install must point at this repo.** The `onpolicy` package is an editable
   install; make sure it resolves to `MAC_COPY` (not an old copy). From the repo root:
   ```bash
   pip install -e . --no-deps
   ```
2. **OpenMP workaround.** Set `KMP_DUPLICATE_LIB_OK=TRUE` to avoid the
   `OMP: Error #15` crash on Windows.
   - PowerShell: `$env:KMP_DUPLICATE_LIB_OK="TRUE"`

### Training + evaluation command (the canonical run)

```bash
python onpolicy/scripts/train/train_mpe.py \
  --env_name MPE \
  --scenario_name simple_spread \
  --algorithm_name mappo \
  --seed 1 \
  --disable_messages \
  --use_eval \
  --eval_interval 5 \
  --eval_disable_messages \
  --eval_noise_std 0.25
```

This trains MAPPO on `simple_spread` and, every 5 episodes, runs evaluation including the
causal-influence (CIC) measurement.

**wandb note:** the script prompts for wandb on first run. To skip the prompt entirely,
add `--use_wandb` (this flag is `store_false`, so passing it **disables** wandb). Omit it
if you want wandb logging (choose offline mode at the prompt).

### Useful flags

- `--disable_messages` — zero messages during **training** (causal ablation of training).
- `--eval_disable_messages` — add the zero-message reward eval pass.
- `--eval_noise_std 0.1` — add the noisy-message reward eval pass.
- `--eval_causal_influence` — `store_false`, default **on**. Pass it to **disable** the
  CIC measurement.
- `--num_agents`, `--episode_length`, `--num_env_steps`, `--n_rollout_threads`,
  `--n_eval_rollout_threads`, `--experiment_name` — standard run controls.

---

## 6. Where Results Are Stored & In What Format

All runs write under:

```
onpolicy/scripts/results/MPE/simple_spread/mappo/<experiment_name>/run<N>/
```

| Output | Location | Format |
|--------|----------|--------|
| **Causal-influence metrics (primary)** | `<run_dir>/causal_influence.csv` | CSV, one row per eval |
| Console summary | stdout (`[EVAL RESULTS]` block) | text, every eval |
| Scalar logs | `<run_dir>/logs/` (TensorBoard) **or** wandb if enabled | TensorBoard event files / wandb |
| Saved models | `<run_dir>/models/` | PyTorch checkpoints |
| Raw message tensors (debug) | `<run_dir>/messages/episode_*_*.npy` | NumPy arrays |

### `causal_influence.csv` schema

One header row, then one row per evaluation:

```
total_num_steps, causal_influence_kl_agent0, causal_influence_kl_agent1,
causal_influence_kl_mean, causal_influence_value_sensitivity_agent0,
causal_influence_value_sensitivity_agent1, causal_influence_value_sensitivity_mean
```

Example:
```
total_num_steps,causal_influence_kl_agent0,causal_influence_kl_agent1,causal_influence_kl_mean,causal_influence_value_sensitivity_agent0,causal_influence_value_sensitivity_agent1,causal_influence_value_sensitivity_mean
25,0.00446,0.00347,0.00397,0.09165,0.37878,0.23522
150,0.04825,0.02832,0.03829,0.72965,0.48782,0.60873
```

This CSV is the recommended source for analysis/plots (opens directly in pandas/Excel).

---

## 7. End-to-End Data Flow

**Training (gradient-coupled communication):**
```
prev_share_obs [batch, num_agents*obs_dim]
  -> reshape [batch, num_agents, obs_dim]
  -> message_head -> softmax -> token_embedding        (gradients enabled)
  -> aggregate (exclude self) -> messages [batch, message_dim]
  -> concat(obs, messages) -> actor.base -> action_logits
  -> PPO loss.backward() -> gradients reach message_head & token_embedding

aggregated_messages (rollout)
  -> attention aggregation (learnable attention_weight)
  -> concat(cent_obs, messages) -> critic -> value
```

**Causal-influence measurement (eval):**
```
at each step, for the same visited state:
  dist_real  = policy.get_action_distribution(obs, messages=real)
  dist_zero  = policy.get_action_distribution(obs, messages=0)
  KL = KL(dist_real || dist_zero)                      -> policy causal effect

  V_real = critic(cent_obs, messages=real)
  V_zero = critic(cent_obs, messages=0)
  |V_real - V_zero|                                    -> value causal effect
trajectory advances using the real (non-intervened) policy
```

---

## 8. Notes for Future Phases

- **Phase 2 (degradation detection):** threshold on a drop in `causal_influence_kl_mean`
  / `causal_influence_value_sensitivity_mean` relative to a rolling baseline (or on the
  normal-vs-ablated reward gap shrinking). The CSV time series is the input signal.
- **Phase 3 (repair):** start with the cheapest lever — re-anneal the message entropy /
  re-init `message_head` for the affected agent — triggered by the Phase 2 signal.
- **Phase 4 (meta-causal adaptation):** meta-train the repair controller across a
  distribution of perturbations (varying `eval_noise_std`, agent dropout, vocabulary
  corruption).
