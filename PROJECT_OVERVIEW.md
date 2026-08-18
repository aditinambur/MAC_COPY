# Project Overview — Causal Mechanistic Emergent Language (MAC_COPY)

This is the single, authoritative document for the capstone project as it currently
stands in **this repository (`MAC_COPY`)**. It supersedes the older scattered notes
(`COMMUNICATION_CHANGES.md`, `PHASE3_COMPLETE_DOCUMENTATION.md`,
`CAUSAL_EVALUATION_COMPLETE.md`, `EXPERIMENTS.md`). Only the parts that are actually
implemented and verified in this repo are recorded here.

**Last updated:** 2026-08-18. The code is pushed (`618f660` on `origin/aditi`); this file and
`CODE_WALKTHROUGH.md` are held back locally pending review — see §10. The automatic causal-adaptive controller (select → repair →
accept/reject → rollback/escalate) is implemented **and validated end-to-end at the
real repair budget** (15 iterations) — see §5.4–§5.8 and §5.9 for the validated result, §5.10
for the seed-1 control arms, **§5.11 for the held-out validation**, and **§5.12 for the control arms on current
code across two seeds — both open claims now supported at n=2, with the trigger margins and the
retention cost stated honestly**.

---

## 1. Project Aim

**Causal Mechanistic Emergent Language with Online Repair
in Multi-Agent Reinforcement Learning.**

### The one-line framing

> **Use causal diagnostics of communication to decide *when* online repair should be
> applied, in order to recover reward under environment change.**

This gives a clean hierarchy, and it is the framing the whole repo is organised around:

| Layer | Role |
|-------|------|
| **Reward** | the task objective |
| **Communication** | the mechanism being diagnosed |
| **Causal influence (CIC)** | tells you whether communication is still *helping* |
| **Repair** | the intervention, applied only when communication is the likely cause |

The resulting loop is:

1. Agents learn a communication channel.
2. We measure whether that channel *causally* affects behaviour and value.
3. When reward degrades, we test whether communication usefulness degraded too.
4. If yes, we repair the communication pathway online.
5. We keep the repair only if reward **and** communication usefulness recover.

The emphasis is on **causal communication + repair**, not merely maximizing reward.
**All five steps are now implemented and validated end-to-end**, including the accept/
reject decision and the rule-based controller that chooses *which* repair to run (§5.4–
§5.8). What remains is strengthening the *evidence* behind the mechanism — control arms
at real budget, multiple seeds — see §9.

**Scenario / setup this project targets:**

- Environment: **MPE `simple_spread`** (cooperative navigation)
- Algorithm: **`mappo`** (MAPPO; non-recurrent in this configuration)
- Entry point: **`onpolicy/scripts/train/train_mpe.py`**
- Agents: **2** (`--num_agents 2`), landmarks: 3
- Observation dim: 14 per agent; centralized obs dim: 28; message dim = `hidden_size` = 64;
  vocabulary size: 5 tokens.

---

## 2. Roadmap & Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Communication architecture** | Learnable, gradient-trained, discrete-token communication with attention aggregation and a message-aware critic | ✅ Implemented & verified |
| **Phase 1 — Causal Influence of Communication (CIC)** | Intervention-based per-agent measurement of message effect on policy (KL) and value (ΔV) | ✅ Implemented & verified |
| **Phase 2 — Degradation detection** | Detect when communication *benefit* collapses under an environment change, gated on reward AND comm-side signal both | ✅ Implemented & validated (one-shot, offline) — §5.2 |
| **Phase 3 — Online repair** | PPO fine-tune of a selectable slice of the communication pathway in the changed env (`embedding`/`comm`/`full`/`noncomm`) | ✅ Implemented & validated — §5.3 |
| **Phase 3a — Automatic target selection** | Rule-based controller picks the initial repair target from the fingerprint pattern | ✅ Implemented & validated — §5.4 |
| **Phase 3b — Repair acceptance + rollback + escalation** | Keep the repair only if reward *and* comm metrics recover; else rollback and escalate to a stronger target | ✅ Implemented & validated (incl. bit-exact rollback verification) — §5.5–§5.6 |
| **Phase 3c — Experimental control arms** | No-repair, reward-only-trigger, and non-communication-parameter baselines | ✅ Implemented; all four arms run at real budget on seed 1 — §5.7–§5.8, results in §5.10 |
| **Phase 3d — Held-out validation** | Re-derive the recovery for an accepted repair on a disjoint CRN block, so the accept decision is checked against episodes it never saw | ✅ Implemented; 3 confirms + 1 rejection at real budget — §5.11, §5.12 |
| **Phase 3e — Multi-agent robustness** | Same pipeline across four independently trained agents and both code versions | ✅ Run; succeeds whenever the perturbation costs ≈30%+ of reward — §5.12 |
| **Phase 3f — Control arms on current code** | `noncomm` and `reward_only` against the causal controller, on two independently trained agents | ✅ Run; both claims supported at n=2, trigger margins razor-thin — §5.12 |

---

## 3. Files to Look At (this project only)

The project runs through the **`mappo` → `r_mappo`** algorithm stack.

### Entry points & configuration
- **`onpolicy/scripts/train/train_mpe.py`** — training/eval entry point; builds envs,
  runner, and `run_dir`.
- **`onpolicy/scripts/phase2_3_repair.py`** — 🆕 standalone Phase 2 + Phase 3 pipeline
  (load checkpoint → baseline → mirror → detect → repair → re-measure).
- **`onpolicy/config.py`** — all command-line flags (see §6).

### Communication architecture (the "causal mechanistic" message layer)
- **`onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py`**
  - `R_Actor` — discrete-token `message_head` + `token_embedding`, message-conditioned
    actor, gradient-coupled message recomputation in `evaluate_actions`,
    and `get_action_distribution` (used by CIC).
  - `R_Critic` — message-aware critic (centralized obs concatenated with aggregated
    messages).
- **`onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py`** — `R_MAPPOPolicy`:
  attention-based message aggregation (`_apply_attention_aggregation`), gradient-coupled
  recomputation (`_recompute_agent_messages_from_prev_share_obs`), the learnable
  `attention_weight` parameter, and message threading through `get_actions`,
  `get_values`, `evaluate_actions`, `get_action_distribution`.
- **`onpolicy/algorithms/r_mappo/r_mappo.py`** — `R_MAPPO` trainer; `ppo_update`
  forwards `prev_share_obs_batch` for gradient flow, and includes `attention_weight`
  in gradient clipping.
- **`onpolicy/algorithms/utils/act.py`** — `ACTLayer.get_distributions()`.
- **`onpolicy/utils/shared_buffer.py`** — stores `aggregated_messages`, provides
  `prev_share_obs` from `feed_forward_generator()`.

### Environment perturbation
- **`onpolicy/envs/mirror_wrapper.py`** — 🆕 `MirrorObsVecEnv` + `compute_mirror_indices`;
  the controlled "environment change" used by Phase 2/3.

### Runner & measurement
- **`onpolicy/runner/shared/mpe_runner.py`** — rollout (`collect`), message aggregation,
  evaluation (`eval`), CIC (`_eval_causal_influence`), CRN-paired intervention eval
  (`_eval_with_intervention`), CSV logging (`_save_causal_influence_csv`), and
  best-checkpoint selection.
- **`onpolicy/runner/shared/base_runner.py`** — `run_dir` setup, tagged/best checkpointing
  (`save(tag=...)`, `save_best()`), `restore()` incl. `attention_weight`.

---

## 4. What Is Implemented — Communication & CIC

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
   - Each receiver learns which senders to weight, instead of plain mean pooling
     (self-messages are masked out with `-inf` before the softmax).

3. **Gradient flow into `message_head` during PPO** — `R_Actor.evaluate_actions` +
   `SharedReplayBuffer`
   - The buffer yields `prev_share_obs` (previous-timestep centralized obs).
   - During the PPO update, messages are **recomputed** from `prev_share_obs` through
     `message_head`/`token_embedding`, so policy-loss gradients reach the communication
     parameters (the stored numpy messages carry no gradient).

4. **Message-aware critic** — `R_Critic`
   - Aggregated messages are concatenated onto `cent_obs` before the value MLP, so the
     value function can credit/penalize communication.
   - In `evaluate_actions` the critic receives `agent_messages.detach()`, so the **value
     loss does not** train the communication pathway — only the policy loss does. This
     keeps "messages are useful" a behavioural claim, not a value-fitting artifact.

5. **Message flow, end to end**
   1. Each agent observes local state.
   2. `message_head` produces token logits.
   3. Softmax over the 5-token vocabulary gives token probabilities.
   4. Those probabilities mix the shared `token_embedding` table into a message vector.
   5. Every receiver gets the **full sender set** (`_build_receiver_message_tensor`),
      and attention collapses it to one incoming message.
   6. Actor consumes `[obs, incoming_message]` to choose actions.
   7. Critic consumes `[cent_obs, incoming_message]` to estimate value.

### 4.2 Phase 1 — Causal Influence of Communication (CIC)

An *intervention-based* measurement; it adds **no new network layer**. During evaluation
the trajectory follows the normal (non-intervened) policy, but at every step the trained
policy is queried on the **same state twice**:

- once with the **real** incoming aggregated message, and
- once with that message **ablated (zeroed)** — the do-operator on the message channel.

Two per-agent quantities are recorded:

- **`causal_influence_kl_agent{i}`** — KL between the agent's action distribution with vs.
  without its incoming message. How much the **policy** causally depends on communication.
- **`causal_influence_value_sensitivity_agent{i}`** — `|V_real − V_ablated|`. How much the
  **critic's value estimate** causally depends on communication.

Plus the across-agent means `causal_influence_kl_mean` and
`causal_influence_value_sensitivity_mean`.

**Implementation:** `MPERunner._eval_causal_influence()`, supported by
`R_MAPPOPolicy.get_action_distribution()`, `R_Actor.get_action_distribution()`, and
`ACTLayer.get_distributions()`. Run automatically inside `eval()` (on by default).

**Verified behavior:** causal influence grows with training — in `phase2_3_seed1/run1`
(2M steps) mean KL rose ~0.007 → ~1.38, its highest value being the final eval, while mean
value-sensitivity stayed around 0.35. The strongest agent measured (`phase2_3_seed3/run1`
@2982400) reaches KL 1.86 with `comm_effect` +534.9. That is the trend Phase 2 thresholds on.

> **Note on the reward-level ablations.** `--eval_disable_messages` and `--eval_noise_std`
> run reward-only eval passes (`eval_normal_rewards` / `eval_no_message_rewards` /
> `eval_noisy_message_rewards`), **CRN-paired** (same initial layouts per condition). These
> are coarse reward-level robustness checks; the rigorous per-agent causal signal is the
> CIC above. `eval_comm_effect_vs_no_message` is the CRN-paired *benefit* of communication
> and is the primary Phase 2 trigger.

---

## 5. What Is Implemented — Phase 2 & Phase 3

All of this lives in **`onpolicy/scripts/phase2_3_repair.py`** (standalone, deliberately
*not* wired into the training loop).

### 5.1 The environment change (`mirror_wrapper.py`)

A **one-sided mirror**: selected observation indices are negated on every `reset()`/`step()`,
while the actions the policy emits are still applied in the **true world frame**. A *full*
mirror (obs + action) is a symmetry `simple_spread` is invariant to, so it would not degrade
a good policy — that would defeat the test.

`compute_mirror_indices(num_agents, num_landmarks, scope=...)` supports three scopes:

| scope | what it negates | effect |
|-------|-----------------|--------|
| `all` | x-component of every spatial sub-vector | **too severe** — navigation itself collapses (reward −1.7k → −45k); comm-only repair cannot recover |
| `partner` | x of the other-agent relative position | navigation intact, partner belief partly corrupted |
| **`partner_full`** ✅ | **both** components of the other-agent relative position | navigation intact, partner belief 180°-flipped → coordination/communication degrades *specifically* |

**`partner_full` is the right perturbation for the repair story** (verified: comm_effect
+627 → −711, value_sens 0.47 → 0.26, reward only −1713 → −2822).

### 5.2 Phase 2 — the fingerprint & detector

`RepairRunner.measure_comm_metrics()` collects a **communication fingerprint** over
`--measure_episodes` CRN-paired episodes:

| field | meaning |
|-------|---------|
| `reward` | mean return under the real policy |
| `no_msg_reward` | mean return with incoming messages zeroed |
| `comm_effect` | `reward − no_msg_reward` — the CRN-paired *benefit* of communication |
| `comm_effect_std` | per-episode spread of that benefit |
| `value_sensitivity` | CIC `|V_real − V_ablated|` |
| `kl` | CIC action-distribution KL |

This deliberately separates **"are we doing well?"** (`reward`) from **"are messages
helping?"** (`comm_effect`) from **"is the policy/critic still *sensitive* to messages?"**
(`kl`, `value_sensitivity`).

`detect_degradation()` flags a change only when **BOTH** of these hold (this AND-gate was
added after review; the earlier version triggered on the comm-side signal alone, which
would fire even if reward hadn't actually gotten worse):

```
reward_degraded = (baseline.reward − current.reward) / |baseline.reward| >= reward_drop_ratio_threshold   # default 0.15

comm_threshold = max(baseline.comm_effect − k_sigma * baseline.comm_effect_std,
                     min_ratio * baseline.comm_effect)
comm_related = (current.comm_effect < comm_threshold)
            or (current.value_sensitivity < 0.5 * baseline.value_sensitivity)

degraded = reward_degraded AND comm_related
```

Reward down with comm_effect stable does **not** trigger (not a communication problem);
nor does comm_effect dipping with reward essentially unchanged (nothing to repair) — repair
only fires when communication is the *likely cause* of the reward drop, matching the
project's framing exactly. **KL is intentionally *not* part of the comm-side signal:**
under the mirror the policy still reacts to messages, so KL can stay high even when
messages have stopped *helping*.

`--controller reward_only` swaps this detector for `detect_degradation_reward_only()`,
which triggers on `reward_drop_ratio >= 0.30` alone, **never** reading `comm_effect` /
`value_sensitivity` / `kl` — an explicit baseline for testing whether the comm-aware
trigger above actually earns its keep (§5.7).

### 5.3 Phase 3 — online repair

`RepairRunner.repair()` runs `--repair_iters` PPO update iterations on the **mirrored**
environment. Messages are forced **on** during repair rollouts (`disable_messages=False`)
so the policy re-learns to use communication under the new observation semantics. It logs
the exact trainable-parameter count for whichever target is active, e.g.
`trainable actor params=395  trainable attention params=128` for `comm` — verified against
hand-calculation for every target (§5.3 table).

`_set_repair_trainable(target)` selects the repair scope:

| `--repair_target` | trainable | params (verified) | interpretation |
|-------------------|-----------|--------------------|----------------|
| `embedding` | `token_embedding` only | 320 | keep *what is said* fixed; relearn what the tokens **mean**. The purest thesis mechanism. |
| `comm` | `message_head` + `token_embedding` + `attention_weight` | 395 actor + 128 attn | relearn what to say and what it means. **Does NOT learn "who to trust"** — see the num_agents==2 caveat below. |
| `full` | entire actor + `attention_weight` | 10192 actor + 128 attn | broad policy repair, not just communication repair |
| `noncomm` 🆕 | `act.action_out` only (control arm, §5.8) | 325 | deliberately excludes everything communication-related |

The **critic is always trainable** in every mode — if the environment changes, the value
function must recalibrate for PPO to be stable. It is the value function, not part of the
communication channel.

> ⚠️ **`attention_weight` is mathematically inert with `num_agents=2`.** In
> `_apply_attention_aggregation`, every receiver has exactly one non-self sender, and
> softmax over a single unmasked class is always 1.0 regardless of that class's logit —
> so the aggregated message doesn't depend on `attention_weight`'s value, and its gradient
> is exactly zero. `comm`/`full` repair still mark it trainable (for forward-compatibility
> with >2 agents) but it learns nothing in the current 2-agent configuration. The pipeline
> prints this as a `[NOTE]` at startup so it's never silently overclaimed. **Do not describe
> `comm` repair as repairing "who to trust" while `num_agents==2`.**

> ℹ️ **Accepted repairs are persisted; rejected ones are not.** `_save_accepted_repair()`
> calls `runner.save(tag="accepted_<target>")` on the accept branch only, writing to
> `<run_dir>/models/checkpoint_accepted_<target>/`. A rejected attempt is rolled back and
> never written to disk. See §9.1. *(This box previously claimed the script never calls
> `save()` — that was true of the first implementation pass and is no longer the case.)*

### 5.4 The automatic controller — `select_repair_target()`

A deterministic, hand-written rule (explicitly **not** a trained/learned decision-maker)
that picks the initial repair target from the fingerprint pattern, so target selection is
no longer a manually-typed CLI flag:

```
if reward_drop_ratio >= severe_reward_ratio (0.50):        target = 'full'
elif value_sens_ratio <= sharp_value_sens_ratio (0.50):     target = 'embedding'
else:                                                        target = 'comm'
```

Both thresholds are CLI-configurable (`--select_severe_reward_ratio`,
`--select_sharp_value_sens_ratio`) and explicitly **not claimed to be optimal** — they're
a documented starting point for §9.3's control-arm work, not a tuned result.

Runs automatically whenever `--repair_target` is **omitted** (the new default). Pass
`--repair_target` explicitly to bypass the controller entirely and force one fixed target
— this is also how the `noncomm` control arm (§5.8) is invoked.

### 5.5 Acceptance — `accept_repair()`

```
recovery = (repaired − degraded) / max(ε, baseline − degraded)
accept = reward_recovery >= 0.30  AND  comm_recovery >= 0.30      (both --accept_* configurable)
```

`value_sensitivity` and `kl` stay **diagnostic only, not gating** — real runs show they do
not necessarily move in the same direction as reward/comm_effect during repair (e.g. the
validated run in §5.9's accepted `full` attempt had `value_sensitivity` end up *below*
baseline despite comm_effect recovering 53%). Gating on them too would reject repairs that
otherwise satisfy the actual thesis claim (reward + comm usefulness recovered).

An accepted repair is additionally re-checked on a **disjoint** CRN block
(`--holdout_seed_offset`, default +1000): all three points — baseline, degraded, repaired —
are re-measured on layouts that had no part in the accept decision, and the same arithmetic
is re-run on them. Reported as `HELD-OUT CONFIRMS` / `HELD-OUT DOES NOT CONFIRM` and logged
to `--results_log`; diagnostic only, it never overrides the decision. This exists because the
decision block also defines `baseline`/`degraded` and the escalation ladder may try up to
three targets against it, so an accepted repair could otherwise be one that merely suits those
particular layouts.

Recovery is only well-defined when the metric actually dropped. If
`baseline − degraded <= 0` for a metric (possible for `comm_effect` when detection fires via
the value-sensitivity branch), that metric's recovery is reported as `None` and is **skipped
as a gate** rather than divided by a clamped epsilon — see `accept_repair`'s
`_recovery_ratio` helper.

### 5.6 Rollback and escalation — `run_causal_adaptive_repair()`

The closed loop: select → snapshot → repair → measure → accept/reject → if rejected,
roll back and escalate to the next-stronger target → repeat.

- **Escalation only ever moves upward** through `embedding → comm → full` from wherever
  the controller's initial pick landed — never back down. If `full` is rejected, or every
  attempt in the ladder is exhausted, parameters are restored to the pre-repair snapshot
  and the run reports no accepted repair.
- **Snapshot/rollback covers actor + critic + `attention_weight` + both Adam optimizers'
  state** (`_snapshot_repairable_state` / `_restore_repairable_state`). The optimizer state
  matters: without restoring Adam's momentum buffers too, a rejected attempt's momentum
  would bias the very first gradient steps of the *next* escalation attempt.
- **Empirically verified, not just assumed correct:** a standalone check mutated the
  policy via a real `full`-target repair (reward moved −1918.9→−2152.3), then rolled back,
  and every one of 5 fingerprint metrics matched the pre-repair values to **6 decimal
  places**. Re-verified independently (2026-08-17) at a different measurement point: after a
  real 2-iteration `full` repair changed 9/13 actor tensors, 6/10 critic tensors and
  populated both optimizers (0 → 30 and 18 state tensors), rollback restored all of them
  **bit-identically** and all 6 fingerprint metrics returned to **exactly** their pre-repair
  values (max abs difference `0.0`).
- **The snapshot also covers `trainer.value_normalizer`.** With `--use_valuenorm` on (the
  default), the running return statistics live on the *trainer*, not inside
  `critic.state_dict()`, and are updated on every `ppo_update`. They are snapshotted and
  restored alongside the other five items, for the same reason as the Adam buffers: otherwise
  a rejected attempt's return statistics would carry into the next escalation attempt and bias
  both the critic's regression target and the advantages that weight the policy loss.

### 5.7 Control arm — reward-only trigger (`--controller reward_only`)

Tests the project's central claim directly: does communication-aware detection actually
beat the obvious naive alternative of "repair whenever reward drops"? Triggers purely on
`reward_drop_ratio >= 0.30`; the repair itself uses a **fixed** `comm` target, single
attempt, no escalation, so the *only* thing that differs from the causal controller is the
trigger decision. Verified: correctly fired at drop ratio 0.59 with 395 trainable params,
same as the causal controller's `comm` target.

### 5.8 Control arm — non-communication repair (`--repair_target noncomm`)

Trains only `act.action_out` (325 params: 320 weight + 5 bias) — the actor's final
action-decoding layer, deliberately outside the communication pathway, roughly comparable
in size to `comm`'s 395. Tests whether recovery is actually attributable to repairing
communication, or whether any equally-sized fine-tune would do about as well. Bypasses the
controller/escalation (manual mode, like any explicit `--repair_target`).

### 5.9 ⭐ Validated real-budget result — current code (2026-08-18)

**Every result in this document is from an agent trained with the current code.** Runs on the
pre-2026-08-16 communication code (`phase2_3/run4`) are retired: they predate the `.detach()`
into the critic, the attention-based gradient path and the eval-aggregation fix, so their
numbers are not comparable. Their logs sit in `results/archive_run4_oldcode/` and are cited
nowhere below.

Full pipeline, no manual overrides, real budget (`--repair_iters 15`, `--measure_episodes 6`,
`n_rollout_threads 32`, `partner_full`), on `phase2_3_seed1/run1` @1958400:

| | baseline | degraded | repaired (`comm`) |
|---|---|---|---|
| reward | −1683.1 | −2187.2 | **−1891.4** |
| comm_effect | +326.0 | −370.3 | **−74.4** |
| reward recovery | — | — | **58.7%** |
| comm recovery | — | — | **42.5%** |
| decision | — | — | **ACCEPT** |

The controller selected `comm` automatically, accepted on the first attempt with no escalation,
and the result survived held-out validation on a disjoint seed block (§5.11). `comm_effect`
recovered 295.8 of the 696.3 lost.

Two further agents also repaired successfully, both requiring escalation to `full`:

| agent | `comm_effect` | drop | decision block | held-out |
|---|---|---|---|---|
| `seed3/run1` @2982400 | +534.9 | 0.298 | 68.9% / 47.6% | **122.8% / 70.0%** confirms |
| `seed3/run1` @2918400 | +402.5 | 0.241 | **99.7% / 73.5%** | **173.9% / 99.4%** confirms |

### 5.10 ⭐ Control arms on current code — three agents

Each agent was run as three arms against the same checkpoint: the causal controller, the
`noncomm` control (§5.8) and the `reward_only` trigger (§5.7).

| agent | drop | causal | `noncomm` | `reward_only` |
|---|---|---|---|---|
| `seed1/run1` @1958400 | 0.2995 | ✅ `comm` 58.7/42.5 → **35.3/34.5 confirms** | ⚠️ accepted 40.5/44.5 → **−2.7/65.6 no-confirm** | ❌ silent (margin 0.0005) |
| `seed3/run1` @2982400 | 0.2981 | ✅ `full` 68.9/47.6 → **122.8/70.0 confirms** | ❌ **rejected** 18.0/44.5 | ❌ silent (margin 0.0019) |
| `seed3/run1` @2918400 | 0.2406 | ✅ `full` 99.7/73.5 → **173.9/99.4 confirms** | ⚠️ accepted 37.3/107.3 → **−50.9/148.2 no-confirm** | ❌ silent (**margin 0.0594**) |

**Claim 1 — recovery is communication-specific. Supported three times.** The communication-pathway
repair recovered and confirmed on all three agents. `noncomm` — a same-sized fine-tune of
`act.action_out`, outside the communication pathway — was rejected outright once and passed its
decision block twice only to **fail held-out validation both times**, at −2.7% and −50.9% reward
recovery. Retraining any small slice does not reproduce the result.

> Two of the three `noncomm` failures are invisible without held-out validation — they looked
> like successes on the episodes that judged them. §5.11 is what makes this claim standable.

**Claim 2 — causal triggering beats reward-only. Supported three times, once decisively.** In all
three cases the causal detector fired and produced a confirmed repair while `reward_only` did
nothing. Two of the margins are negligible (0.0005, 0.0019 — the same edge case twice, not
independent evidence). **The third is decisive: at a drop of 0.2406 the naive trigger misses by
0.0594 and the resulting repair recovered 99.7% / 73.5%, confirmed held-out at 173.9% / 99.4%.**
That is a case no reviewer can call a coin flip.

**Retention cost is real and scales with repair breadth.**

| repair | retention on the unperturbed task |
|---|---|
| `full` on @2982400 | **−17.3%** — past the 10% warning threshold |
| `comm` on @1958400 | −9.6% |
| `full` on @2918400 | −9.5% |
| forced repairs on weak agents | −0.4% / +0.4% |

Recovering in the changed environment costs performance in the original one. This is a measured
tradeoff, not a defect — and it is exactly what a reward-only acceptance criterion would hide.

### 5.11 ⭐ Held-out validation

The accept decision is made on `--measure_episodes` CRN layouts, and those same layouts also
define `baseline` and `degraded`, with the escalation ladder free to try up to three targets
against them. An accepted repair could therefore be one that merely suits those layouts. After
any acceptance the pipeline re-derives the whole recovery calculation on a **disjoint** seed
block (`--holdout_seed_offset`, default +1000), re-measuring all three points — it temporarily
restores the pre-repair weights for baseline/degraded, then puts the repaired ones back.
Reported as `HELD-OUT CONFIRMS` / `DOES NOT CONFIRM`; diagnostic only, it never overrides the
decision.

**It discriminates rather than rubber-stamping.** On current code: **3 confirms** (every causal
arm) and **3 rejections** (`noncomm` twice, plus the forced repair on a weak agent). Critically,
those rejections **changed conclusions** — in two cases `noncomm` had already passed its decision
block, so without this check the specificity claim in §5.10 would have read as a tie.

### 5.12 ⭐ What predicts a successful repair

Across every current-code agent measured, the predictor is **how much reward the perturbation
destroys** — not the seed, and not `comm_effect` alone.

| agent | `comm_effect` | drop | outcome |
|---|---|---|---|
| `seed3/run1` @2982400 | +534.9 | 0.298 | ✅ accepted (`full`), confirms |
| `seed1/run1` @1958400 | +326.0 | 0.300 | ✅ accepted (`comm`), confirms |
| `seed3/run1` @2918400 | +402.5 | **0.241** | ✅ accepted (`full`), confirms |
| `seed1/run1` @1990400 | +320.3 | 0.160 | ❌ `comm` −26.1%, `full` 32.5/9.6 — both rejected |
| `seed3/run1` @2502400 | +139.1 | 0.080 | detector declined; forced → both rejected |
| `seed2/run1` @1766400 | +138.9 | 0.093 | detector declined; forced → held-out failed |

**The success threshold lies between 0.16 and 0.24**, not at 0.30 as an earlier reading of fewer
points suggested. That matters: it sits *inside* the 0.15–0.30 band where the causal and naive
triggers disagree, so the two claims in §5.10 do not compete — there is room for the causal
trigger to fire, the naive one to stay silent, and the repair to succeed comfortably.

`comm_effect` alone does **not** predict the outcome: @1990400 (+320.3) and @1958400 (+326.0)
have near-identical communication benefit and opposite results. Nor does the drop predict *which*
target suffices — @1958400 succeeded with `comm` while both seed-3 agents needed `full`.

**The detector's selectivity is exercised.** It declined both weak agents (drop 0.08–0.09) — the
first runs in which `reward_degraded` was ever False, so §5.2's AND-gate claim has now been
tested on real data. Forcing past it produced a non-generalizing fix (seed 2) or none at all
(seed 3).

**Strongest single mechanistic result.** Seed 3's forced `full` attempt recovered reward *above
baseline* (−1728.5 vs −1797.8, 150.1%) while restoring only **18.2%** of `comm_effect`, and was
rejected. Reward recovery is not communication repair, and the acceptance rule tells them apart.

**Training variance: 1 weak run in 3.** Best measured `comm_effect` per run: seed 3 **+534.9**,
seed 1 **+326.0**, seed 2 **+116.9**. Only seed 2 is genuinely weak — its four tested checkpoints
top out at +116.9 and one *improved* under the mirror (drop −0.04).

**The checkpoint-selection score is unreliable — it failed in two of three runs**, each time
burying the run's best communicator: `seed1/run1` chose step **70400** (3.5% into training, after
`value_sensitivity` spiked to 2.019 at a single eval and the score weights it 100×);
`seed3/run1` chose step 2502400 (+139.1) over its own final 2982400 (**+534.9**). It is a greedy
running maximum over 60–90 noisy evals, so an early fluke wins permanently. **Any claim that a
seed produced a weak agent is unsafe if it rests on `checkpoint_best`.** Sweep the top-KL
checkpoints with `--no_repair` and choose on measured `comm_effect` and drop — see
`results/newseed1_checkpoint_sweep.jsonl` and `results/seed23_checkpoint_sweep.jsonl`.

> **Caveats when quoting.** Held-out reward recoveries above 100% (122.8%, 173.9%) mean the
> repaired agent beat its own baseline on those layouts — check the denominator before citing
> them. `noncomm`'s two gates are genuinely independent (its `no_msg_reward` moves), so its
> comm-recovery figures can rise partly by making the no-message policy worse, which is why they
> reach 107–148% while reward recovery goes negative. And `noncomm` (325 params) was compared
> against `full` successes (10,192 params) on two of three agents — not size-matched; no
> similarly-sized non-communication target exists.

**Logs** (all current code): `seed3mid_{causal,noncomm,rewardonly}.jsonl`,
`seed3best_{causal,noncomm,rewardonly}.jsonl`, `newseed1_{ck1958400,noncomm,rewardonly}.jsonl`,
`newseed1_agent.jsonl`, `seed2_agent*.jsonl`, `seed3_agent*.jsonl`,
`newseed1_checkpoint_sweep.jsonl`, `seed23_checkpoint_sweep.jsonl`. `results/` is gitignored.

**Still open.** n=3 on both claims, same direction each time — but no statistics yet, and all
three agents come from two training runs. §9.3 stands.

---

## 6. How to Run

### Prerequisites (one-time, Windows + conda)

0. **Activate the `marl` conda env first — this is not optional.** Everything below assumes
   it. `conda activate marl`.
1. **Editable install must point at this repo:**
   ```bash
   pip install -e . --no-deps
   ```
2. **OpenMP workaround** — set `KMP_DUPLICATE_LIB_OK=TRUE` to avoid `OMP: Error #15`.
   - PowerShell: `$env:KMP_DUPLICATE_LIB_OK="TRUE"`

**Verified environment (re-checked 2026-08-17):** the `marl` env is **Python 3.9.25, torch
2.8.0+cpu, gym 0.21.0, numpy 2.0.2**. *(An earlier version of this doc recorded torch
2.7.1+cpu / gym 0.17.2 / numpy 2.2.6 — those are not the versions this code currently runs
under.)* The `base` env will not work: its editable `onpolicy` install points at a different
repo, and its Python 3.12 lacks the `imp` module that `onpolicy/envs/mpe/scenarios/__init__.py`
imports.

### 6.1 Training a communication-reliant checkpoint

```bash
python onpolicy/scripts/train/train_mpe.py \
  --env_name MPE \
  --scenario_name simple_spread \
  --algorithm_name mappo \
  --seed 1 \
  --num_agents 2 \
  --num_env_steps 2000000 \
  --use_eval \
  --eval_interval 5 \
  --eval_disable_messages \
  --experiment_name phase2_3 \
  --use_wandb
```

> ⚠️ **`--use_wandb` is required, and it *disables* wandb** (`store_false`, default True).
> Without it `train_mpe.py` takes the wandb branch and `save_dir` becomes `wandb.run.dir`, so
> every checkpoint lands in `<experiment_name>/wandb/offline-run-<timestamp>/` instead of a
> numbered `run<N>/` folder. Everything still trains correctly — it is just written somewhere
> awkward to find. Pass it.
>
> Two other flags that are easy to omit and expensive to get wrong: **`--num_env_steps`**
> defaults to **10e6**, five times the 2M this project uses (`episodes = num_env_steps //
> episode_length // n_rollout_threads`, so 2M → 312 episodes → ~38 min; 10M → ~3 h), and
> **`--experiment_name`** defaults to `check`, which mixes a new run in with the superseded
> ones. For a second seed use e.g. `--seed 2 --experiment_name phase2_3_seed2`, which lands at
> `.../mappo/phase2_3_seed2/run1/`.

**Best-checkpoint output.** Whenever an eval beats the running-best score
(`reward + max(0, comm_effect) + 100 × value_sensitivity`), `save_best()` writes:

| path | meaning |
|------|---------|
| `<run_dir>/models/checkpoint_best/` | **always the current best.** Pass this straight to `--model_dir` |
| `<run_dir>/models/checkpoint_best_<steps>/` | one folder per new best, pruned to the newest `--best_keep` (default 3) |
| `<run_dir>/models/best_*.pt` | flat files, kept for backwards compatibility |

The two folder forms use the exact filenames `restore()` expects (`actor.pt` / `critic.pt` /
`attention_weight.pt`), so **`checkpoint_best/` can be handed to Phase 2/3 directly** — no need
to hunt through `causal_influence.csv` for the best step. The flat `best_*.pt` files still
**cannot** be loaded by `restore()` (§7.1). Runs made before this was added have only the flat files and must be
addressed by an explicit `checkpoint_<steps>/` directory.

> 🚨 **Do NOT pass `--disable_messages` when training a checkpoint for Phase 2/3.**
> With it, messages are zeroed during rollout, so the behaviour policy never acts on real
> messages — communication is learned only indirectly through the `ppo_update`
> gradient-recompute path, giving weak, high-variance comm (a 2M-step seed-1 run produced
> kl ≈ 0.007 and comm_effect swamped by ±190 noise). Removing it makes the policy condition
> on real aggregated messages during rollout → genuine causal communication.
> **Keep `--eval_disable_messages`** — that only *adds* the no-message eval pass used to
> measure `comm_effect`.

Note `use_linear_lr_decay` is **off** by default, so a shorter `--num_env_steps` does not
distort the LR schedule.

**Checkpoint acceptance criteria** (≥8 CRN eval episodes, normal env):

| metric | threshold |
|--------|-----------|
| `eval_normal_reward` | ≳ −950 |
| `eval_comm_effect_vs_no_message` | ≥ +120 |
| `causal_influence_value_sensitivity_mean` | ≥ 0.40 |

### 6.2 Phase 2 detect-only

```bash
python onpolicy/scripts/phase2_3_repair.py \
  --env_name MPE --scenario_name simple_spread --algorithm_name mappo --seed 1 \
  --model_dir "onpolicy/scripts/results/MPE/simple_spread/mappo/phase2_3_seed1/run1/models/checkpoint_1958400" \
  --eval_disable_messages \
  --measure_episodes 6 \
  --mirror_scope partner_full \
  --no_repair
```

### 6.3 Phase 3 — automatic causal controller (recommended default)

```bash
python onpolicy/scripts/phase2_3_repair.py \
  --env_name MPE --scenario_name simple_spread --algorithm_name mappo --seed 1 \
  --model_dir "onpolicy/scripts/results/MPE/simple_spread/mappo/phase2_3_seed1/run1/models/checkpoint_1958400" \
  --measure_episodes 6 \
  --mirror_scope partner_full \
  --repair_iters 15
```

No `--repair_target` → `select_repair_target()` auto-picks, and `run_causal_adaptive_repair()`
handles accept/reject/rollback/escalation automatically (§5.4–§5.6). This is the exact
command validated end-to-end in §5.9 (with `--n_rollout_threads 1 --n_eval_rollout_threads 1`
added for a faster/reproducible single-threaded run — see the caveat in §5.9 before
comparing its numbers to earlier 32-thread runs).

### 6.4 Manual target / control arms

```bash
# Force one fixed target, no controller, no escalation:
... --repair_target comm     # or embedding / full

# Non-communication control arm (Priority 8):
... --repair_target noncomm

# Reward-only trigger baseline (Priority 7) — tests the comm-aware detector's value-add:
... --controller reward_only
```

Swap `--repair_target embedding` / `comm` / `full` / `noncomm` to compare repair scopes
manually; add `--controller reward_only` to compare against the naive trigger.

> ✅ **Fixed (was: `run_dir = Path(--model_dir)`, littering the checkpoint directory).**
> `run_dir` now defaults to a **sibling** folder next to `--model_dir` — never inside it —
> named `<checkpoint_dir_name>_repair_runs/<controller>_<target>_seed<N>_<timestamp>/`, so
> `--model_dir`'s own `actor.pt`/`critic.pt`/`attention_weight.pt` stay pristine no matter
> how many repair runs point at the same checkpoint. Pass `--run_dir <path>` to override.
> Verified: ran end-to-end, confirmed no new files appear under `--model_dir` itself, and
> the accepted repair's weights landed at
> `<run_dir>/models/checkpoint_accepted_<target>/{actor,critic,attention_weight}.pt`.

### 6.5 Useful flags

| flag | effect |
|------|--------|
| `--disable_messages` | zero messages during **training** (see the warning in §6.1) |
| `--eval_disable_messages` | add the zero-message reward eval pass during **training-time** `eval()`. ⚠️ It is a **no-op inside `phase2_3_repair.py`** — that pipeline calls `measure_comm_metrics()` directly, which always runs both conditions regardless of this flag |
| `--eval_noise_std 0.25` | add the noisy-message reward eval pass |
| `--eval_causal_influence` | `store_false`, default **on**; pass it to **disable** CIC |
| `--save_messages` | dump per-step message tensors to `<run_dir>/messages/*.npy`. **Off by default** — writes 2 files per env-step and cripples long runs |
| `--use_wandb` | `store_false`, so passing it **disables** wandb. **Required for training** — without it checkpoints go to `wandb.run.dir`, not `run<N>/models/` (§6.1) |
| `--best_keep` | 🆕 sliding-window size for `checkpoint_best_<steps>/` folders (default 3; `0` keeps all) — §6.1 |
| `--num_env_steps` | defaults to **10e6**; this project uses **2000000** (~38 min at 32 threads) |
| `--experiment_name` | defaults to `check`; always set it (e.g. `phase2_3_seed2`) so runs don't collide |
| `--mirror_scope` | `all` / `partner` / `partner_full` |
| `--measure_episodes` | CRN-paired episodes per condition per fingerprint |
| `--repair_target` | 🆕 default **`None`** (auto-selected); pass explicitly for `embedding`/`comm`/`full`/`noncomm` manual mode |
| `--repair_iters` | PPO iterations during repair |
| `--detect_k_sigma`, `--detect_min_ratio` | comm-side detector sensitivity |
| `--detect_reward_drop_ratio` | 🆕 reward-side detector gate (default 0.15) — see §5.2 |
| `--no_repair` | detect only, skip repair (kept as the no-repair experimental control) |
| `--controller` | 🆕 `causal` (default) or `reward_only` — §5.7 |
| `--select_severe_reward_ratio`, `--select_sharp_value_sens_ratio` | 🆕 `select_repair_target` thresholds — §5.4 |
| `--accept_reward_recovery`, `--accept_comm_recovery` | 🆕 `accept_repair` thresholds — §5.5 |
| `--reward_only_drop_ratio` | 🆕 `--controller reward_only` trigger threshold — §5.7 |
| `--results_log <path>` | 🆕 append one JSON line per run for later multi-seed aggregation (no sweep is run automatically) |
| `--run_dir <path>` | 🆕 override the default sibling output folder — §9.1 |
| `--n_eval_rollout_threads` | must be **1**; `phase2_3_repair.py` refuses to start otherwise, because CRN seeding only works in-process. Override with `--allow_unpaired_eval` (invalidates `comm_effect`) |
| `--n_rollout_threads` | repair rollout width. Changing it makes runs incomparable — §5.9 caveat |
| `--allow_unpaired_eval` | 🆕 escape hatch for the check above. Only for debugging; results are not CRN-paired |
| `--holdout_episodes` | 🆕 held-out re-check of an **accepted** repair on a disjoint seed block. `-1` (default) = same count as `--measure_episodes`; `0` disables. Diagnostic only — never changes the accept decision |
| `--holdout_seed_offset` | 🆕 offset for the held-out block (default 1000). Must be ≥ `--measure_episodes` or the run refuses to start |

---

## 7. Results, Checkpoints & Storage

All runs write under:

```
onpolicy/scripts/results/MPE/simple_spread/mappo/<experiment_name>/run<N>/
```

| Output | Location | Format |
|--------|----------|--------|
| **Causal-influence metrics (primary)** | `<run_dir>/causal_influence.csv` | CSV, one row per eval |
| Console summary | stdout (`[EVAL RESULTS]` block) | text, every eval |
| Scalar logs | `<run_dir>/logs/` (TensorBoard) or wandb | event files / wandb |
| Rolling checkpoint | `<run_dir>/models/{actor,critic,attention_weight}.pt` | PyTorch |
| **Named snapshots** | `<run_dir>/models/checkpoint_<steps>/` | PyTorch, one per eval |
| **Best snapshot (loadable)** | `<run_dir>/models/checkpoint_best/` | PyTorch — always the current best; pass directly to `--model_dir` |
| Best snapshot history | `<run_dir>/models/checkpoint_best_<steps>/` | PyTorch, newest `--best_keep` kept (default 3) |
| Best snapshot (flat, legacy) | `<run_dir>/models/best_*.pt` | PyTorch — **not** loadable by `restore()` |
| Raw message tensors (debug) | `<run_dir>/messages/*.npy` | NumPy, only with `--save_messages` |

### 7.1 ⭐ The canonical checkpoints (current code)

```
onpolicy/scripts/results/MPE/simple_spread/mappo/phase2_3_seed1/run1/models/checkpoint_1958400/
onpolicy/scripts/results/MPE/simple_spread/mappo/phase2_3_seed3/run1/models/checkpoint_2982400/
onpolicy/scripts/results/MPE/simple_spread/mappo/phase2_3_seed3/run1/models/checkpoint_2918400/
```

All three repair successfully and confirm held-out (§5.9–§5.10). Use `@1958400` as the primary
reference — it is the one that succeeds with the narrow `comm` target rather than needing `full`.

| checkpoint | `comm_effect` | drop | KL | repair |
|---|---|---|---|---|
| `seed1/run1` @1958400 | +326.0 | 0.300 | 0.96 | `comm`, confirms |
| `seed3/run1` @2982400 | +534.9 | 0.298 | 1.86 | `full`, confirms |
| `seed3/run1` @2918400 | +402.5 | 0.241 | 1.67 | `full`, confirms |

> ⚠️ **Do not use `checkpoint_best/` without checking which step it resolves to.** The selection
> score failed in two of three runs, each time burying the run's best communicator (§5.12). None
> of the three checkpoints above was the one its run selected. Sweep the top-KL checkpoints with
> `--no_repair` and pick on measured `comm_effect` and reward drop instead.

> **Weak agents, for reference:** `seed2/run1` tops out at `comm_effect` +116.9 and
> `seed3/run1` @2502400 measures +139.1. Neither is damaged enough by the perturbation for the
> detector to fire (drop 0.08–0.09), so neither is usable for repair experiments.

### 7.2 Run inventory

| run | steps | code | verdict |
|-----|-------|------|---------|
| `phase2_3_seed3/run1` | 3.0M | current | ⭐ strongest agent (+534.9 @2982400); its `checkpoint_best` is **not** the one to use |
| `phase2_3_seed1/run1` | 2.0M | current | ⭐ primary reference (@1958400); `checkpoint_best` resolves to step 70400 — unusable |
| `phase2_3_seed2/run1` | 2.0M | current | weak run — tops out at +116.9, detector never fires |
| `phase2_3/run4` | 2.0M | **pre-2026-08-16** | retired — predates the comm-layer fixes; results archived, not cited |
| `phase2_3/run1`, `run3` | — | old | debris / superseded |
| `crn_fix/run1`, `check/wandb/*` | — | old | superseded |

> **Path gotcha.** Checkpoints live under `models/` with an **underscore**:
> `run1/models/checkpoint_1958400`, *not* `run1/checkpoint1958400`. Pointing at a non-existent
> path silently creates empty `logs/`+`models/` folders there and then fails in `restore()`.

### 7.3 `causal_influence.csv` schema

```
total_num_steps, causal_influence_kl_agent0, causal_influence_kl_agent1,
causal_influence_kl_mean, causal_influence_value_sensitivity_agent0,
causal_influence_value_sensitivity_agent1, causal_influence_value_sensitivity_mean
```

One header row, then one row per evaluation (append mode). This CSV is the recommended
source for analysis/plots.

---

## 8. End-to-End Data Flow

**Training (gradient-coupled communication):**
```
prev_share_obs [batch, num_agents*obs_dim]
  -> reshape [batch, num_agents, obs_dim]
  -> message_head -> softmax -> token_embedding        (gradients enabled)
  -> attention aggregate (self masked out) -> messages [batch, message_dim]
  -> concat(obs, messages) -> actor.base -> action_logits
  -> PPO policy loss.backward() -> gradients reach message_head, token_embedding,
     attention_weight
  -> critic receives messages.detach()                 (value loss does NOT train comm)
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

**Phase 2/3 pipeline:**
```
load checkpoint
  -> measure fingerprint on NORMAL env            (baseline)
  -> wrap eval_envs in MirrorObsVecEnv
  -> measure fingerprint on MIRRORED env          (degraded)
  -> detect_degradation(baseline, degraded)
  -> if degraded: wrap train envs, PPO fine-tune selected comm params
  -> measure fingerprint on MIRRORED env          (repaired)
  -> report recovery as % of lost comm benefit
```

---

## 9. What Is Left

The full repair loop (select → repair → accept/reject → rollback/escalate) is **built and
validated at real budget** (§5.4–§5.9). What's left is the evidence layer that makes the
mechanism's numbers defensible, plus two smaller polish items. Ordered by leverage.

### 9.1 Repair plumbing — ✅ done (2026-08-16)

- **Persist repaired weights.** `_save_accepted_repair()` now calls `runner.save(tag=...)`
  whenever a repair is accepted (never for rejected attempts), writing to
  `<run_dir>/models/checkpoint_accepted_<target>/{actor,critic,attention_weight}.pt`.
  Verified on disk, not just by log line.
- **`--run_dir` override**, defaulting to a sibling folder next to `--model_dir` — verified
  no litter lands inside the checkpoint directory anymore (§6.4).
- **`onpolicy/scripts/aggregate_repair_results.py`** (new file) — reads one or more
  `--results_log` JSONL files, groups by `(controller, target, n_rollout_threads,
  mirror_scope)`, prints n / acceptance-rate / mean±std recovery per group, writes a CSV,
  and optionally runs a Welch's t-test between two named groups. Verified against real
  JSONL data from this session (correctly computed 50% acceptance rate and matching
  mean/std across 2 real rows; `--compare` and CSV output both confirmed working).
- **Debris folders in `run1/`/`run3/`** — still not cleaned up (unrelated to this fix,
  low priority).

### 9.2 Re-run the §5.9 validation at production `n_rollout_threads` — ✅ done (2026-08-16)

Confirmed: at matched `n_rollout_threads=32`, the causal controller auto-selects `comm`,
accepts it on the first attempt (no escalation needed), and reproduces the manual
reference numbers almost to the decimal (§5.9's updated caveat). The thread-count
mismatch, not a controller defect, fully explains the `n_rollout_threads=1` run's
different (escalated-to-`full`) outcome. **Timed at 9 min 29 s** for one single-attempt
run — the basis for the timing estimates in §9.3 below.

### 9.3 Control arms at real budget — seed 1 done, multi-seed still needed

**Update (2026-08-16, later same day):** all four arms now have a real-budget (seed 1)
data point — see §5.10 for the full table. What that unblocked and what's still open:

- **`reward_only` vs. `causal` head-to-head** — done for seed 1: they tied exactly (same
  detection decision, same repair, bit-identical outcome). That is **not yet evidence for
  the project's central claim** — it means this perturbation is severe enough that both
  detectors always agree, so the comparison hasn't actually been exercised yet. Needs
  either more seeds (hoping for one where the two thresholds disagree) or a deliberately
  milder perturbation.
- **`noncomm` at real budget** — done for seed 1: `comm` beat `noncomm` on both metrics
  (see §5.10 for the current-code arms), but the gap is small relative to the measured noise
  (`comm_effect_std` ~260-360). Suggestive, not yet conclusive.
- **No-repair control at real budget** — done for seed 1, confirms the degraded fingerprint
  is stable and reproducible (bit-identical to the causal run's degraded numbers).
- **Seeds.** Every number in §5.10 is a single seed. Repeat over ≥3–5 seeds with error
  bars before either finding above is reportable — `comm_effect_std` is large relative to
  the effect (e.g. ±340 against a baseline comm_effect of ~400), so single-seed
  differences this size cannot yet be distinguished from noise.

### 9.4 Detector upgrades

- **Rolling online detector.** `detect_degradation()` is a one-shot offline
  baseline-vs-current comparison. Make it a rolling-window detector over the
  `causal_influence.csv` time series so it can fire *during* a run.
- **Per-step CIC logging** in `mpe_runner` so you can see *which* messages/timesteps lost
  effect, instead of one episode-mean number.
- **Statistical honesty.** With `--measure_episodes 6–8`, `comm_effect_std` is estimated
  from very few samples and the `k_sigma` band is correspondingly fragile.

### 9.5 Generality

Everything is validated on one perturbation (`partner_full` mirror), one scenario
(`simple_spread`), 2 agents. At minimum add a second perturbation type (agent dropout or
vocabulary corruption) to show the detector is perturbation-agnostic — the repair
mechanism already is. Going to 3+ agents would also make the `comm`/`full` targets' claim
about `attention_weight` real again (it's currently inert at exactly 2 agents — §5.3).

### 9.6 Threshold sensitivity — none of the 8 cutoffs are validated

Every decision point in the pipeline is gated by a threshold picked as a reasonable round
number, never tuned or tested against data:

| threshold | flag | current value | gates |
|---|---|---|---|
| comm-side sigma band | `--detect_k_sigma` | 2.0 | `detect_degradation`'s comm-side signal |
| comm-side ratio band | `--detect_min_ratio` | 0.5 | `detect_degradation`'s comm-side signal |
| reward-side gate | `--detect_reward_drop_ratio` | 0.15 | `detect_degradation`'s reward AND-gate |
| escalate-to-`full` cutoff | `--select_severe_reward_ratio` | 0.50 | `select_repair_target` |
| start-with-`embedding` cutoff | `--select_sharp_value_sens_ratio` | 0.50 | `select_repair_target` |
| reward acceptance bar | `--accept_reward_recovery` | 0.30 | `accept_repair` |
| comm acceptance bar | `--accept_comm_recovery` | 0.30 | `accept_repair` |
| naive-trigger cutoff | `--reward_only_drop_ratio` | 0.30 | `detect_degradation_reward_only` |

This is not hypothetical risk: on the one real seed-1 run (§5.10), `reward_drop_ratio`
came out at **0.473** — 3 percentage points from the 0.50 `select_severe_reward_ratio`
cutoff that decides whether the controller tries `comm` first or jumps straight to `full`.
A few points either way and the entire "controller starts minimal, escalates only when
needed" narrative for that run would not exist. **Once §9.3's multi-seed runs exist**,
re-check each cutoff against the spread of `reward_drop_ratio`/`value_sens_ratio` values
actually observed across seeds — if real results cluster near a cutoff, that cutoff is
reporting a coin flip, not a decision, and needs to move or be justified explicitly.

---

## 10. Delta vs. the Last Push

Last pushed commit: **`618f660`** — *"feat: causal-adaptive communication repair pipeline
(Phase 2/3)"* on `origin/aditi` (which now tracks; the old `origin/study-notes` target is gone).
That commit carried all the code below, plus `TEAM_HANDOFF.md`.

**Still uncommitted:** this file (`PROJECT_OVERVIEW.md`), `CODE_WALKTHROUGH.md`, and `memory/`.
Also note `results/` is gitignored, so the experiment logs cited throughout §5.9–§5.12 exist only
locally — add a `!results/*.jsonl` exception before anyone else needs to reproduce those tables.

The historical delta below is kept for reference.

```
 onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py |  79 ++++++++++---
 onpolicy/algorithms/r_mappo/r_mappo.py                |   9 +-
 onpolicy/config.py                                    |   6 +-
 onpolicy/runner/shared/base_runner.py                 |  42 +++++--
 onpolicy/runner/shared/mpe_runner.py                  | 123 ++++++++++++++-----
 5 files changed, 198 insertions(+), 61 deletions(-)
```

### 10.1 New files (untracked)

| file | purpose |
|------|---------|
| `onpolicy/envs/mirror_wrapper.py` | `MirrorObsVecEnv` + `compute_mirror_indices` — the controlled environment change (3 scopes) |
| `onpolicy/scripts/phase2_3_repair.py` | the whole Phase 2/3 pipeline **plus the automatic causal-adaptive controller** (324 → 731 lines across two implementation passes) — see below |
| `memory/` | project notes (`MEMORY.md`, `phase2-3-direction.md`) |

**`phase2_3_repair.py`, second pass — the controller (2026-08-16):** on top of the original
`measure_comm_metrics` / `_set_repair_trainable` / `repair` / `detect_degradation`:
- `detect_degradation()` gained a reward-drop AND-gate (§5.2) — was comm-side-only before.
- `_set_repair_trainable()` gained the `'noncomm'` target and an honest docstring on
  `attention_weight`'s inertness at `num_agents==2`.
- `repair()`'s try/finally was tightened so setup failures (not just rollout failures)
  still restore trainability/`disable_messages`; it now logs trainable-parameter counts.
- New: `RepairRunner._snapshot_repairable_state` / `_restore_repairable_state` (actor +
  critic + `attention_weight` + both Adam optimizers' state — §5.6).
- New: `select_repair_target()` (§5.4), `accept_repair()` (§5.5),
  `detect_degradation_reward_only()` (§5.7), `run_causal_adaptive_repair()` (§5.6
  orchestration), `_append_results_log()` / `_log_and_close()` (structured JSON logging).
- `parse_args()` gained `--controller`, `--select_*`, `--accept_*`,
  `--detect_reward_drop_ratio`, `--reward_only_drop_ratio`, `--results_log`; `--repair_target`
  default changed from `'comm'` to `None` (auto-selected) and gained the `'noncomm'` choice.
- `main()` was restructured into three branches: manual override (any explicit
  `--repair_target`), `--controller reward_only`, and the default causal controller.
- Validated: 6 smoke tests (one per target × the two baselines, all at `repair_iters=2`)
  plus one standalone bit-exact rollback verification, plus one real-budget
  (`repair_iters=15`) end-to-end run of the full causal controller — §5.9.

### 10.2 Modified files

**`onpolicy/runner/shared/base_runner.py`**
- `save(tag=None)` — a tag writes a permanent `checkpoint_<tag>/` snapshot instead of
  overwriting the rolling `actor.pt`/`critic.pt`. **This is what fixed checkpoints being
  clobbered every save.**
- `save()` / `save_best()` now also persist **`attention_weight.pt`**, so a restored policy
  reproduces the exact message aggregation it was evaluated with.
- `restore()` now loads `attention_weight.pt` when present.

**`onpolicy/runner/shared/mpe_runner.py`** (largest change)
- **Receiver-aware message tensor.** New `_build_receiver_message_tensor()`; rollout and
  both eval paths now carry `[n_envs*n_agents, n_agents, message_dim]` (the *full sender
  set* per receiver) instead of a pre-averaged `[n_envs*n_agents, message_dim]`.
  This is a **semantic fix**: previously eval hard-coded uniform mean-pooling, bypassing
  the learned `attention_weight` — so eval-time aggregation did not match training.
- New `latest_policy_messages` buffer, honouring `--disable_messages`.
- **Best-checkpoint selection** in `run()`: saves `checkpoint_<steps>/` at every eval, plus
  `best_*.pt` by the score `reward + max(0, comm_effect) + 100 × value_sensitivity`. The
  100× puts value-sensitivity (~O(1)) on the same scale as reward/comm_effect (~O(100s)),
  so a strong-reward-but-weak-comm snapshot cannot win.
- New tracking fields `latest_eval_normal_reward` / `latest_eval_comm_effect` /
  `latest_eval_value_sensitivity`, populated in `eval()`.
- `_eval_causal_influence(crn_seed=None)` — accepts a CRN seed for paired measurement.
- `_log_messages()` now gated behind `--save_messages` (was unconditional and wrote 2 files
  per env-step, which severely slowed long runs).

**`onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py`**
- Extracted `_apply_attention_aggregation()` from the inline branch in
  `_prepare_agent_messages`, and fixed the self-mask to use `masked_fill(..., -inf)`
  instead of the `+ (self_mask - 1) * 1e9` trick. The old einsum
  (`'bij,ji->bi'` with `attention_weight.t()`) is replaced by an explicit
  receiver-indexed `'bsd,bd->bs'`.
- New `_recompute_agent_messages_from_prev_share_obs()` — recomputes messages **through
  attention** during PPO (previously the gradient path did not match rollout aggregation),
  with `nan_to_num` guarding.
- `evaluate_actions` now prefers the recomputed messages when `prev_share_obs` is given.
- Critic receives `agent_messages.detach()` so the **value loss no longer trains the
  communication pathway**.

**`onpolicy/algorithms/r_mappo/r_mappo.py`**
- `attention_weight` is included in actor gradient clipping / grad-norm when it requires
  grad. Previously it was optimized but **excluded from `clip_grad_norm_`**.

**`onpolicy/config.py`**
- Added `--save_messages` (default off).

> The Phase 2/3 CLI flags (`--mirror_scope`, `--measure_episodes`, `--repair_iters`,
> `--repair_target`, `--detect_k_sigma`, `--detect_min_ratio`, `--no_repair`) are defined
> locally in `phase2_3_repair.py`'s `parse_args`, not in `config.py`.

### 10.3 Suggested commit split

1. `fix: match eval-time message aggregation to training (receiver-aware tensor + attention)`
2. `fix: include attention_weight in grad clipping; detach messages into critic`
3. `feat: tagged + best checkpointing with attention_weight persistence`
4. `perf: gate per-step message dumps behind --save_messages`
5. `feat: mirror wrapper + Phase 2/3 detect-and-repair pipeline`
