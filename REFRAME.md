# Reframe — Original Plan vs. What Was Actually Built

This document maps the project's **original challenge analysis** (the three-layer plan: Causal
Mechanistic Layer, Online Repair and Adaptation, Meta-Causal Adaptation) onto **what exists in
this repository today**, with the evidence for each.

It exists because a fair number of the anticipated challenges turned out not to apply, while a
different set consumed most of the actual effort. Writing up the original list unchanged would
misrepresent the work in both directions.

**The single biggest divergence:** the system that was built is **one-shot and offline** —
load a checkpoint, measure a baseline, apply the environment change, detect, repair once, accept
or roll back. It is deliberately not wired into the training loop
(`onpolicy/scripts/phase2_3_repair.py` is standalone). Several planned challenges assumed a
*continuous online* loop and therefore do not arise.

All results cited below come from agents trained with the current code. Earlier runs on the
pre-2026-08-16 communication code are retired and not counted (see `PROJECT_OVERVIEW.md` §5.9).

---

## 4.1 Causal Mechanistic Layer

**What was built.** Intervention-based Causal Influence of Communication (CIC): at each visited
state the trained policy is queried twice — once with the real incoming message, once with that
message zeroed — and two quantities are recorded per agent:

- `causal_influence_kl` — KL between the two action distributions (does the message change
  *behaviour*?)
- `causal_influence_value_sensitivity` — `|V_real − V_ablated|` (does it change the critic's
  *estimate*?)

Separately, `comm_effect = reward − no_msg_reward` measures whether the messages actually make
the team score better (*usefulness*, as distinct from *sensitivity*).

**What was not built:** no Model of Other Agents, no information-bottleneck penalty, no causal-
structure regulariser.

### Challenge 1 — Noisy causal signals ✅ Real, and the central measurement problem

The concern was correct, and it is the hardest measurement issue in the project.
`comm_effect` carries a per-episode spread of **±290 to ±440** against means of **+300 to +535** —
the noise is nearly the size of the effect.

**Solved differently than planned.** The original solutions were smoothing, multiple counterfactual
samples, confidence thresholds and variance reduction. What actually works here is:

- **Common Random Numbers (CRN).** Every condition is forced to start from an identical layout by
  re-seeding before each `reset()`, so the difference between conditions is *paired* rather than a
  difference of two independently-sampled noisy means. This is what makes `comm_effect`
  trustworthy at only 6–8 episodes.
- **Held-out re-validation.** After any accepted repair, the whole recovery calculation is
  re-derived on a disjoint block of episodes that played no part in the decision.

> ⚠️ CRN pairing only works in-process. The vec-env classes expose no `seed()` method, so seeding
> reaches only the parent process; with more than one eval thread the pairing silently disappears.
> The pipeline now refuses to start with `--n_eval_rollout_threads > 1` for this reason.

### Challenge 2 — Dependence on critic / MOA quality ⚠️ Half-applies

There is no MOA, so that half is moot. The critic dependency is real, and it caused a concrete
failure — but not the one anticipated.

`value_sensitivity` proved unstable enough to be actively misleading. In one training run it
spiked to **2.019** at a single evaluation; because the checkpoint-selection score weights it
100×, that one reading won permanently and the run's "best" agent was fixed at **3.5% into
training**. In another run the same score chose an agent measuring `comm_effect` +139.1 over that
run's own final checkpoint at **+534.9**.

**How it is handled now:** `value_sensitivity` and `kl` are **diagnostic only** — they are logged
and reported but gate nothing. Acceptance depends solely on `reward` and `comm_effect`. Checkpoint
selection should not rely on the built-in score (see §4.4.1).

### Challenge 3 — Multi-objective balancing ❌ Does not apply

There is no information-bottleneck penalty and no causal regulariser, so there are no competing
objectives to balance. Training is standard PPO.

The one balancing decision that was made is different in kind: **the critic receives
`agent_messages.detach()`**, so the value loss cannot train the communication parameters. Only the
policy loss shapes them. This keeps "messages are useful" a claim about *behaviour* rather than an
artifact of the critic fitting itself.

Measured effect of that choice: policy-side dependence rose (KL 1.384 vs 1.043 at a matched step)
while value-side sensitivity fell (0.346 vs 0.533) — exactly what removing the critic's influence
over messages predicts. It did **not** weaken communication overall: a current-code agent reaches
`comm_effect` **+534.9**, the strongest measured anywhere in the project.

---

## 4.2 Online Repair and Adaptation

**What was built.** `RepairRunner.repair()` — PPO fine-tuning restricted to a selected parameter
slice, on the changed environment:

| target | trains | params |
|---|---|---|
| `embedding` | `token_embedding` only | 320 |
| `comm` | `message_head` + `token_embedding` (+ `attention_weight`) | 395 |
| `full` | the entire actor | 10,192 |
| `noncomm` | `act.action_out` only — control arm | 325 |

Around it: accept/reject on normalised recovery, bit-exact rollback, and escalation to a stronger
target when an attempt is rejected.

### Challenge 1 — Non-stationarity during repair ❌ Does not apply

The concern was that changing message meanings would leave the partner acting on the old ones.
**Both agents share a single policy** (`share_policy=True`), so they update together — there is no
partner holding a stale interpretation. Incremental updates, synchronisation phases and cooldowns
are unnecessary in this configuration.

*(It would apply in a genuinely decentralised variant with per-agent policies. Worth noting as
future work rather than as a solved problem.)*

### Challenge 2 — Over-repair and oscillation ❌ Does not apply

Repair runs **once**, offline, against a loaded checkpoint. There is no repeated triggering, so
oscillation is not possible and repair thresholds, cooldown timers and repair penalties are not
needed.

### What replaced these: accepting a bad repair ✅ Real, and solved

The actual risk was different — that a repair *looks* successful on the episodes used to judge it
and is not. Two mechanisms address it:

- **Rollback.** A rejected attempt restores actor, critic, `attention_weight`, **both Adam
  optimisers** and the **value normaliser**, bit-exactly. Verified by mutating all of them with a
  real PPO run and confirming all six fingerprint metrics return to their pre-repair values with
  max absolute difference `0.0`.
- **Held-out validation.** Two repairs passed their own acceptance test and then **failed** on
  fresh episodes (−2.7% and −50.9% reward recovery). Without this check they would have been
  recorded as successes.

---

## 4.3 Meta-Causal Adaptation

**What was built.** A deterministic, rule-based detector and target selector — explicitly *not* a
learned meta-controller:

```
reward_degraded = (baseline.reward − current.reward) / |baseline.reward| >= 0.15
comm_related    = comm_effect below its tolerance band  OR  value_sensitivity halved
degraded        = reward_degraded AND comm_related
```

### Challenge 1 — Distinguishing noise from true drift ✅ Real, and solved by the AND-gate

Requiring **both** a reward drop and a communication-side signal is precisely what prevents firing
on noise. A comm_effect dip with reward unchanged means there is nothing worth repairing; a reward
drop with communication intact is not a communication problem.

**Demonstrated on real data.** Two independently trained weak agents showed degraded communication
(`comm_effect` +138.9 → −99.8 and +139.1 → −186.7) while reward barely moved (drops of 9.3% and
8.0%, under the 15% gate). The detector declined both — the first runs in which `reward_degraded`
was ever False.

**And the refusal was correct.** Forcing repair past the detector on those agents produced a fix
that failed held-out validation in one case, and no valid fix at all in the other. That is direct
evidence the gate is doing useful work.

### Challenge 2 — Meta-trigger stability ❌ Does not apply

One-shot, so there is nothing to oscillate. Hysteresis, two-threshold triggers and post-repair
cooldowns become relevant only if this is made continuous.

### Not built — the learned meta-controller

`select_repair_target()` is a hand-written `if/elif/else` on the fingerprint. The originally
planned *learned* controller, meta-trained across a distribution of perturbations, does not exist.

There is a further caveat: the selector has picked `comm` **every single time** it has run. Its
`embedding` branch requires `value_sensitivity` to halve under degradation, and that value has
**risen** in every run measured — so that branch has never once fired. The escalation ladder does
work (`comm` → `full` has fired and succeeded), but the initial selection has never been given a
situation where it would choose differently.

---

## 4.4 Challenges that actually appeared (absent from the original plan)

These consumed most of the real effort.

### 4.4.1 The checkpoint-selection score is unreliable

The score `reward + max(0, comm_effect) + 100 × value_sensitivity` is a greedy running maximum
over 60–90 noisy evaluations, so a single early fluke wins permanently. **It failed in two of
three runs**, each time burying that run's best communicator (step 70400 in one; +139.1 chosen
over +534.9 in the other).

Any statement of the form "seed N produced a weak agent" is unsafe if it rests on
`checkpoint_best`. Current practice: sweep the highest-KL checkpoints with `--no_repair` and
choose on measured `comm_effect` and reward drop.

### 4.4.2 Training variance

Roughly **one training run in three** produces an agent too weak to test on. Best measured
`comm_effect` per run: **+534.9**, **+326.0**, **+116.9**. The checkpoint-selection procedure is
therefore load-bearing and must be reported, not treated as a detail.

### 4.4.3 Overfitting the acceptance decision

The same 6 CRN episodes defined the baseline, defined the degraded state, **and** judged up to
three repair attempts. Addressed by held-out re-validation on a disjoint seed block — which has
since rejected repairs that their own decision block accepted.

### 4.4.4 Repair has a retention cost

Recovering in the changed environment costs performance in the original one, and the cost scales
with how much of the actor is retrained: **−17.3%** for a `full` repair, **−9.5%** and **−9.6%**
for narrower ones. This is a measured tradeoff rather than a defect — and it is exactly what a
reward-only acceptance criterion would hide.

### 4.4.5 Threshold fragility

Several decisions hinge on margins far smaller than the measurement noise. The trigger comparison
twice turned on drop ratios of **0.2995** and **0.2981** against a 0.30 threshold — misses of
0.0005 and 0.0019. A third case at 0.2406 (margin 0.0594) is the one that can be relied on.

---

## Summary

| layer | planned challenges | status |
|---|---|---|
| **4.1 Causal Mechanistic** | noisy signals | ✅ real — solved by CRN pairing + held-out validation |
| | critic / MOA quality | ⚠️ no MOA exists; critic issue real, handled by making `value_sensitivity` non-gating |
| | multi-objective balancing | ❌ no competing objectives; the `detach()` is the one design choice |
| **4.2 Online Repair** | non-stationarity | ❌ shared policy, both agents update together |
| | over-repair / oscillation | ❌ one-shot, not continuous |
| | *(actual)* accepting a bad repair | ✅ solved by bit-exact rollback + held-out validation |
| **4.3 Meta-Causal** | noise vs drift | ✅ solved by the AND-gate; demonstrated declining two weak agents |
| | trigger stability | ❌ one-shot |
| | learned controller | ❌ not built — rule-based stand-in, and it has never picked anything but `comm` |
| **4.4 Unplanned** | checkpoint selection, training variance, acceptance overfitting, retention cost, threshold fragility | ⚠️ these were the real work |

**What this means for the write-up.** The causal-measurement and detection layers are built and
have been exercised on real data. The repair layer works and has been validated on held-out
episodes. The meta-causal layer is a rule-based stand-in whose selector has not yet been shown to
discriminate. Several originally-anticipated difficulties do not apply to a one-shot offline
design and should not be claimed as solved — the honest framing is that the design avoids them,
not that it overcame them.
