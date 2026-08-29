# Implementation Plan — Degradation Detection (Phase 2) and Online Repair (Phase 3)

This plan details the technical specifications, mathematical formulation, and architecture changes required to implement online degradation detection and repair mechanisms for emergent communication in MPE.

---

## 1. Goal Description

In multi-agent reinforcement learning (MARL), emergent communication protocols are fragile. When the environment undergoes a perturbation (e.g., communication channel noise, sensory shift, or agent drops), the causal effect of messages on agent policies degrades, leading to cooperative failure. 

This change introduces:
1.  **Phase 2 — Degradation Detection**: A rolling-baseline detector monitoring the Causal Influence of Communication (CIC).
2.  **Phase 3 — Online Repair Mechanism**: Dynamic recovery triggers (entropy re-annealing, message head re-initialization, or vocab expansion) to restore emergent communication online.

---

## 2. Technical Specification & Mathematical Formulation

### 2.1 Phase 2: Degradation Detection
Let $C_t \in \mathbb{R}$ represent the mean Causal Influence of Communication (KL divergence) evaluated at training step $t$:
$$C_t = \frac{1}{N} \sum_{i=0}^{N-1} D_{KL} \left( P(a_i \mid o_i, m_i) \parallel Q(a_i \mid o_i, 0) \right)$$

We maintain an Exponential Moving Average (EMA) rolling baseline $\mu_t$ representing the "healthy" communication baseline:
$$\mu_t = (1 - \alpha) \mu_{t-1} + \alpha C_t$$
where $\alpha \in (0, 1]$ is the smoothing factor (default: $\alpha = 0.2$).

During evaluation, if a perturbation (e.g., message noise $\sigma > 0$ via `--eval_noise_std`) is applied, we measure the perturbed causal influence $C_{eval}$. A degradation flag $D_t \in \{0, 1\}$ is triggered if the current influence drops below a threshold fraction $\theta$ of the baseline:
$$D_t = \mathbb{I} \left( C_{eval} < \theta \cdot \mu_t \right)$$
where $\theta \in (0, 1)$ is the degradation threshold (default: $\theta = 0.70$, indicating a $30\%$ drop in causal influence).

### 2.2 Phase 3: Online Repair Levers
When $D_t = 1$, the runner triggers one of three configurable online repair mechanisms:

1.  **Entropy Re-annealing**:
    *   Temporarily multiply the policy/message entropy coefficient $\beta$ by a boost factor (e.g., $\beta \leftarrow 5 \times \beta$) to force agents to explore alternative communication tokens.
2.  **Partial Re-initialization**:
    *   Re-initialize the weights of the `message_head` linear layer of the affected agents using orthogonal initialization:
        $$\text{message\_head.weight} \sim \mathcal{O}(gain=1.0)$$
    *   This breaks local minima while preserving the actor's navigation policy weights.
3.  **Vocabulary Expansion**:
    *   Dynamically increase the vocabulary size $V \leftarrow V + k$ (e.g., from 5 to 8).
    *   Resize the `token_embedding` weights matrix, copying old embeddings over and randomly initializing the new slots.

---

## 3. Alternative Approaches (Pros & Cons)

| Detection / Repair Method | Pros | Cons |
| :--- | :--- | :--- |
| **CIC-KL Detection (Proposed)** | * Directly measures how much policy actions rely on communication. * Fast to calculate on the fly. | * Does not measure if the communication is actually *beneficial* (could have high KL but lead to bad actions). |
| **Normal vs. Ablated Reward Gap Detection** | * Directly measures cooperative value of communication. * Simple to implement. | * High variance in reward makes it slow to detect degradation compared to action distribution shifts. |
| **Mutual Information (MI) Tracking** | * Standard information-theoretic measure. * Captures non-linear dependencies. | * Extremely expensive to compute online (requires density estimation of joint states). |
| **Entropy Re-annealing (Proposed Repair)** | * Easy to implement by scaling the PPO trainer loss coefficient. * Preserves learned coordination. | * Exploration takes several training updates to converge. |
| **Partial Re-init (Proposed Repair)** | * Instantly breaks deadlocks. * Resets communication without erasing navigation skills. | * Can cause temporary performance instability right after re-initialization. |
| **Meta-Causal Adaptation (Phase 4)** | * Learns optimal repair policies across multiple perturbations. * High generalization. | * Complex meta-gradient updates (MAML-style) requiring massive computation. |

---

## 4. Research Studies & Academic Context

Our proposed detection and repair architecture is backed by key research in emergent communication:

1.  **Seth Karten's Thesis (CMU, 2023)**: *Emergent Communication and Decision-Making in Multi-Agent Teams*. Highlights that cooperative MARL agents develop high mutual dependency on communication channels, but these protocols catastrophically break under domain shifts. Karten proposes intervention-based measurements (similar to CIC) to track channel decay.
2.  **Karten et al. (2023)**: *On the role of emergent communication for social learning in multi-agent reinforcement learning*. Demonstrates that communication helps coordinate navigation in MPE, but adding environmental noise deteriorates the semantic agreement of tokens.
3.  **Lowe et al. (2017)**: *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* (MADDPG). Validates the centralized critic design, confirming that value estimations must incorporate communication vectors to maintain accurate credit assignment during shifts.

---

## 5. Proposed Code Changes

### Component: Runner & Metrics

#### [MODIFY] [mpe_runner.py](file:///d:/capstone/capstone_project/MAC_COPY/onpolicy/runner/shared/mpe_runner.py)
*   Add rolling EMA variables (`self.kl_baseline`, `self.val_baseline`) to `__init__`.
*   Update `eval()` to compute rolling averages and check if the current eval step triggers `degradation_status = "degraded"` ($D_t = 1$).
*   Save the degradation status flag inside `causal_influence.csv` (schema update).

### Component: Training & Neural Networks

#### [MODIFY] [r_mappo.py](file:///d:/capstone/capstone_project/MAC_COPY/onpolicy/algorithms/r_mappo/r_mappo.py)
*   Add a method `trigger_repair(repair_type: str)` to execute:
    *   **Entropy Boost**: Increase `self.entropy_coef` temporarily.
    *   **Message Head Reset**: Call policy weights re-initialization.
    *   **Vocab Resize**: Resize policy embedding layers.

#### [MODIFY] [r_actor_critic.py](file:///d:/capstone/capstone_project/MAC_COPY/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py)
*   Add `reinit_message_head()` and `expand_vocabulary(new_vocab_size)` helper functions.

### Component: Control API & Dashboard

#### [MODIFY] [main.py](file:///d:/capstone/capstone_project/MAC_COPY/api/main.py)
*   Add support in `list_runs()` and `/api/runs/{run_id}/metrics` to parse the new `degradation_status` column.
*   Add a `POST /api/runs/repair` endpoint to manually trigger a repair lever during active training for interactive debugging.

#### [MODIFY] [App.tsx](file:///d:/capstone/capstone_project/MAC_COPY/frontend/src/App.tsx)
*   Render a **System Status Badge** (Healthy / Degraded) on the Dashboard header.
*   Add a **"Trigger Repair Lever"** selection dropdown (Entropy Boost, Re-init Head, Expand Vocab) next to the Stop Subprocess button.

---

## 6. Verification Plan

### Automated Verification
*   Execute tests running training with `eval_noise_std 0.5` to verify that degradation is detected within 10 episodes and the CSV logs `degradation_status` properly.
*   Trigger `POST /api/runs/repair` programmatically and assert that the entropy coefficient updates or weights change.

### Manual Verification
1. Boot the stack using `docker-compose up --build`.
2. Configure a training run with `--eval_noise_std 0.25`.
3. Monitor the Dashboard and watch the **System Status Badge** transition from "Healthy" (green) to "Degraded" (red) once the noise deteriorates the causal KL divergence.
4. Click **Trigger Repair (Entropy Boost)** and verify on the charts if the causal influence rebounds.
