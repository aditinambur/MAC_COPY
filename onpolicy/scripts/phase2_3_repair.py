#!/usr/bin/env python
"""
Phase 2 (degradation detection) + Phase 3 (online repair) standalone pipeline.

Flow (the "meta-causal loop"):
  1. Load a trained, communication-reliant policy from --model_dir.
  2. Measure a BASELINE communication fingerprint on the normal environment.
  3. Apply an environment change: a one-sided mirror of the observation x-axis
     (onpolicy/envs/mirror_wrapper.py). Measure the DEGRADED fingerprint.
  4. DETECT whether reward degraded AND communication degraded (both required --
     detect_degradation; --controller reward_only swaps in a reward-only trigger
     baseline for comparison, detect_degradation_reward_only).
  5. If degradation is detected: the causal controller (select_repair_target) picks an
     initial repair target -- 'embedding' / 'comm' / 'full' -- from the fingerprint
     pattern, snapshots the pre-repair state, and runs ONLINE REPAIR (a few PPO
     iterations on the changed environment, restricted to that target's parameters).
  6. ACCEPT the repair if reward AND comm_effect recovered enough (accept_repair);
     otherwise ROLL BACK to the snapshot and ESCALATE to the next-stronger target
     (run_causal_adaptive_repair), until accepted or the ladder is exhausted.
  Pass --repair_target explicitly to bypass the controller/escalation and force a
  single fixed-target attempt -- also how the 'noncomm' non-communication control
  arm (Priority 8) is run.

The "fingerprint" is measured with the same intervention machinery used during training
eval (MPERunner._eval_with_intervention / _eval_causal_influence), so the numbers are
directly comparable to the causal_influence.csv produced during training.

Example (causal controller, auto target selection + escalation):
  python onpolicy/scripts/phase2_3_repair.py \
    --env_name MPE --scenario_name simple_spread --algorithm_name mappo --seed 1 \
    --model_dir onpolicy/scripts/results/MPE/simple_spread/mappo/phase2_3/run4/models/checkpoint_1990400 \
    --measure_episodes 8 --repair_iters 15

Example (reward-only trigger baseline, for comparison):
  ...same flags... --controller reward_only

Example (non-communication control arm):
  ...same flags... --repair_target noncomm
"""

import sys
import os
import time
import numpy as np
import torch
from pathlib import Path

from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.env_wrappers import DummyVecEnv, SubprocVecEnv
from onpolicy.envs.mirror_wrapper import (
    MirrorObsVecEnv,
    PartnerRotationObsVecEnv,
    compute_mirror_indices,
)
from onpolicy.runner.shared.mpe_runner import MPERunner


def _make_mpe_vec_env(all_args, n_threads, seed_base):
    """Build a (Dummy/Subproc) vectorized MPE env, matching train_mpe.py construction."""
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "MPE":
                raise NotImplementedError("phase2_3_repair only supports MPE")
            env = MPEEnv(all_args)
            env.seed(seed_base + rank * 1000)
            return env
        return init_env
    if n_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    return SubprocVecEnv([get_env_fn(i) for i in range(n_threads)])

def _wrap_perturbed_env(venv, all_args, mirror_idx):
    """Apply either the legacy mirror or Task-8 rotation perturbation."""
    if all_args.perturb_angle is None:
        return MirrorObsVecEnv(venv, mirror_idx)

    return PartnerRotationObsVecEnv(
        venv,
        num_agents=all_args.num_agents,
        num_landmarks=all_args.num_landmarks,
        angle_deg=all_args.perturb_angle,
    )

class RepairRunner(MPERunner):
    """MPERunner extended with Phase 2 measurement/detection and Phase 3 online repair."""

    # -- Phase 2: measurement -------------------------------------------------
    @torch.no_grad()
    def measure_comm_metrics(self, n_episodes, base_seed):
        """
        Measure the communication fingerprint on the CURRENT self.eval_envs:
          - reward            : mean return under the real (non-intervened) policy
          - no_msg_reward     : mean return with the incoming message ablated (zeroed)
          - comm_effect       : reward - no_msg_reward  (CRN-paired benefit of communication)
          - value_sensitivity : |V_real - V_ablated| averaged over the trajectory (CIC)
          - kl                : KL(action_dist_real || action_dist_ablated) (CIC)

        PRIORITY 1C (comparability check, documented not changed): all three quantities for
        episode k are measured with the SAME crn_seed, so they start from the identical initial
        layout and use the identical deterministic-action policy. They are still two distinct
        interventions by design, not the same measurement twice: no_msg_reward lets the
        "no message" counterfactual drive the WHOLE trajectory (a full-episode do-operator, so
        it can diverge from the real trajectory after step 0), while value_sensitivity/kl ablate
        the message only to compute the paired forward pass at each visited state and then
        advance using the REAL policy (trajectory never altered -- see _eval_causal_influence).
        That difference is intentional (it's what makes CIC a same-state sensitivity measure
        rather than a behavioral outcome measure); only the CRN seeding is meant to be shared.
        """
        normal, no_msg, value_sens, kl_vals = [], [], [], []
        for k in range(n_episodes):
            crn_seed = base_seed + k
            normal.append(float(np.sum(self._eval_with_intervention('normal', crn_seed=crn_seed))))
            no_msg.append(float(np.sum(self._eval_with_intervention('no_messages', crn_seed=crn_seed))))
            cic = self._eval_causal_influence(crn_seed=crn_seed)
            value_sens.append(float(cic['causal_influence_value_sensitivity_mean'][0]))
            kl_vals.append(float(cic['causal_influence_kl_mean'][0]))
        normal = np.array(normal)
        no_msg = np.array(no_msg)
        eff = normal - no_msg
        return {
            'reward': float(normal.mean()),
            'no_msg_reward': float(no_msg.mean()),
            'comm_effect': float(eff.mean()),
            'comm_effect_std': float(eff.std()),
            'value_sensitivity': float(np.mean(value_sens)),
            'kl': float(np.mean(kl_vals)),
        }

    # -- Phase 3: online repair ----------------------------------------------
    def _set_repair_trainable(self, target):
        """
        Choose which parameters the online repair is allowed to update:
          - 'embedding': only token_embedding -- re-learn the *meaning* of each message token
                         (the semantic vectors), keeping who-says-what (message_head), the
                         message aggregation, and the rest of the policy fixed. This is the
                         purest "repair the message embeddings' meaning" mechanism.
          - 'comm':      message_head + token_embedding + attention_weight -- repair what to
                         say and what it means. attention_weight is included for
                         forward-compatibility with >2 agents; with exactly 2 agents (the only
                         configuration this repo currently trains) it does NOT learn "who to
                         trust" -- see the num_agents==2 note printed at startup and in
                         PRIORITY-1B below. Do not describe 'comm' repair as repairing trust
                         weighting while num_agents==2.
          - 'full':      the entire actor (+ attention_weight) -- a full policy re-train.
          - 'noncomm':   PRIORITY 8 control arm. Trains ONLY act.action_out (the actor's final
                         action-decoding linear layer), which is NOT part of the communication
                         pathway, and leaves attention_weight untouched. Roughly comparable in
                         parameter count to 'comm' (both are small: a few hundred parameters
                         each -- the exact counts are logged below so "comparable-sized" is
                         checkable, not assumed). Used to test whether recovery from repair is
                         actually attributable to repairing communication, or whether generic
                         fine-tuning of an equally-sized, non-communication actor slice recovers
                         just as much.
        The critic is always trainable so value estimates stay calibrated for stable PPO; it
        is the value function, not part of the communication channel.
        """
        actor = self.trainer.policy.actor
        if target == 'full':
            for p in actor.parameters():
                p.requires_grad = True
        else:
            keys = {'embedding': ('token_embedding',),
                    'comm': ('message_head', 'token_embedding'),
                    'noncomm': ('action_out',)}[target]
            for name, p in actor.named_parameters():
                p.requires_grad = any(k in name for k in keys)
        for p in self.trainer.policy.critic.parameters():
            p.requires_grad = True
        if getattr(self.trainer.policy, "attention_weight", None) is not None:
            self.trainer.policy.attention_weight.requires_grad = (target in ('comm', 'full'))

    def _restore_all_trainable(self):
        for p in self.trainer.policy.actor.parameters():
            p.requires_grad = True
        for p in self.trainer.policy.critic.parameters():
            p.requires_grad = True
        if getattr(self.trainer.policy, "attention_weight", None) is not None:
            self.trainer.policy.attention_weight.requires_grad = True

    # -- PRIORITY 4: snapshot / rollback --------------------------------------
    def _snapshot_repairable_state(self):
        """
        Deep-copy everything ANY repair target (up to and including 'full') can modify:
        the whole actor, the critic (always trainable during repair, so it can drift too),
        attention_weight, AND both optimizers' state_dicts (Adam's per-parameter exp_avg /
        exp_avg_sq momentum buffers). Without the optimizer snapshot, a rejected attempt's
        momentum would carry into the next escalation attempt and bias its very first
        gradient steps toward whatever direction just got rejected -- rollback would restore
        the WEIGHTS but the optimizer would not actually be starting fresh. Always snapshotting
        the superset -- rather than only the currently-selected target's slice -- keeps
        rollback correct regardless of which target ends up being tried.

        The same argument applies to trainer.value_normalizer, which is why it is captured too.
        With --use_valuenorm (on by default) it is a standalone ValueNorm module owned by the
        TRAINER, holding running_mean / running_mean_sq / debiasing_term -- so it is NOT inside
        critic.state_dict() and would otherwise survive a rollback. It is updated on every
        ppo_update via cal_value_loss, and it is not inert: it rescales the critic's regression
        target and feeds `advantages = returns - denormalize(value_preds)`, the weight on every
        action in the policy loss. Under --use_popart the normalizer IS critic.v_out, already
        covered by the critic snapshot; deep-copying its state_dict again is harmless.
        """
        import copy
        attn = getattr(self.trainer.policy, "attention_weight", None)
        value_normalizer = getattr(self.trainer, "value_normalizer", None)
        return {
            'actor': copy.deepcopy(self.trainer.policy.actor.state_dict()),
            'critic': copy.deepcopy(self.trainer.policy.critic.state_dict()),
            'attention_weight': attn.detach().clone() if attn is not None else None,
            'actor_optimizer': copy.deepcopy(self.trainer.policy.actor_optimizer.state_dict()),
            'critic_optimizer': copy.deepcopy(self.trainer.policy.critic_optimizer.state_dict()),
            'value_normalizer': (copy.deepcopy(value_normalizer.state_dict())
                                 if value_normalizer is not None else None),
        }

    def _restore_repairable_state(self, snapshot):
        """Restore exactly the parameters + optimizer state captured by _snapshot_repairable_state."""
        self.trainer.policy.actor.load_state_dict(snapshot['actor'])
        self.trainer.policy.critic.load_state_dict(snapshot['critic'])
        attn = getattr(self.trainer.policy, "attention_weight", None)
        if attn is not None and snapshot['attention_weight'] is not None:
            with torch.no_grad():
                attn.copy_(snapshot['attention_weight'])
        self.trainer.policy.actor_optimizer.load_state_dict(snapshot['actor_optimizer'])
        self.trainer.policy.critic_optimizer.load_state_dict(snapshot['critic_optimizer'])
        value_normalizer = getattr(self.trainer, "value_normalizer", None)
        if value_normalizer is not None and snapshot.get('value_normalizer') is not None:
            value_normalizer.load_state_dict(snapshot['value_normalizer'])

    def repair(self, repair_iters, target='comm'):
        """
        Online repair on the CHANGED environment (self.envs must already be mirrored).
        Runs `repair_iters` PPO update iterations. Messages are kept ON during the repair
        rollouts so the policy re-learns to use communication under the new observations.

        PRIORITY 1A: prev_disable is captured, but nothing is mutated, before the try block
        opens, so the finally clause restores trainability + disable_messages even if
        _set_repair_trainable itself (e.g. an unknown target) raises, not just if collect() /
        env.step() / compute() / train() raise.
        """
        prev_disable = self.all_args.disable_messages
        try:
            # Ensure messages are used during repair rollouts (the trained checkpoint may
            # have used --disable_messages during its original training).
            self.all_args.disable_messages = False
            self._set_repair_trainable(target)
            n_trainable = sum(p.numel() for p in self.trainer.policy.actor.parameters() if p.requires_grad)
            attn = getattr(self.trainer.policy, "attention_weight", None)
            n_attn = attn.numel() if (attn is not None and attn.requires_grad) else 0
            print("  [REPAIR] target={}  trainable actor params={}  trainable attention params={}".format(
                target, n_trainable, n_attn))

            self.warmup()
            for it in range(repair_iters):
                for step in range(self.episode_length):
                    values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, reconstructions = self.collect(step)
                    obs, rewards, dones, infos = self.envs.step(actions_env, reconstructions)
                    data = (obs, rewards, dones, infos, values, actions, action_log_probs,
                            rnn_states, rnn_states_critic, reconstructions)
                    self.insert(data)
                self.compute()
                train_infos = self.train()
                print("  [REPAIR] iter {:>2}/{}  policy_loss={:.4f}  value_loss={:.4f}".format(
                    it + 1, repair_iters,
                    float(train_infos.get('policy_loss', np.nan)),
                    float(train_infos.get('value_loss', np.nan))))
        finally:
            self._restore_all_trainable()
            self.all_args.disable_messages = prev_disable


def detect_degradation(baseline, current, k_sigma, min_ratio, reward_drop_ratio_threshold):
    """
    Phase 2 detector. Triggers repair only when BOTH:
      (a) reward has significantly degraded (drop >= reward_drop_ratio_threshold of
          |baseline reward|), AND
      (b) communication effectiveness has degraded (comm_effect below its max-tolerable
          threshold, or value-sensitivity halved).
    Reward down with comm_effect stable does NOT trigger (not a communication problem); nor
    does comm_effect dipping with reward essentially unchanged (nothing to repair). This
    matches the intended design: repair only when communication is the LIKELY CAUSE of the
    reward drop, not merely coincidentally different from baseline. (KL is intentionally NOT
    part of the comm-side signal: under the mirror the policy still reacts to messages, so KL
    can stay high even when messages stop helping -- see PROJECT_OVERVIEW.md.)
    """
    baseline_reward_mag = abs(baseline['reward']) + 1e-6
    reward_drop_ratio = (baseline['reward'] - current['reward']) / baseline_reward_mag
    reward_degraded = reward_drop_ratio >= reward_drop_ratio_threshold

    # Tolerable lower bound on comm benefit: whichever is the *stricter* (higher) of a
    # sigma-band on the baseline and a fixed fraction of the baseline benefit.
    sigma_band = baseline['comm_effect'] - k_sigma * baseline['comm_effect_std']
    ratio_band = min_ratio * baseline['comm_effect']
    comm_threshold = max(sigma_band, ratio_band)

    comm_degraded = current['comm_effect'] < comm_threshold
    value_degraded = current['value_sensitivity'] < 0.5 * baseline['value_sensitivity']
    comm_related = bool(comm_degraded or value_degraded)

    degraded = bool(reward_degraded and comm_related)

    return degraded, {
        'reward_drop_ratio': reward_drop_ratio,
        'reward_drop_ratio_threshold': reward_drop_ratio_threshold,
        'reward_degraded': bool(reward_degraded),
        'comm_threshold': comm_threshold,
        'comm_degraded': bool(comm_degraded),
        'value_degraded': bool(value_degraded),
        'comm_related': comm_related,
        'reward_drop': baseline['reward'] - current['reward'],
        'comm_effect_drop': baseline['comm_effect'] - current['comm_effect'],
    }


def detect_degradation_reward_only(baseline, current, reward_drop_ratio_threshold):
    """
    PRIORITY 7: reward-only trigger baseline. Decides whether to repair using ONLY the drop in
    `reward` relative to its own baseline magnitude -- it deliberately does NOT look at
    comm_effect, value_sensitivity, or kl. Exists to test the paper's core claim: does
    communication-aware triggering (detect_degradation) do better than the obvious naive
    alternative of "repair whenever reward drops"? Must be run as a separate --controller
    reward_only arm, never blended into the causal controller's decision.
    """
    baseline_mag = abs(baseline['reward']) + 1e-6
    reward_drop = baseline['reward'] - current['reward']
    reward_drop_ratio = reward_drop / baseline_mag
    degraded = bool(reward_drop_ratio >= reward_drop_ratio_threshold)
    return degraded, {
        'reward_drop': reward_drop,
        'reward_drop_ratio': reward_drop_ratio,
        'reward_drop_ratio_threshold': reward_drop_ratio_threshold,
    }


def select_repair_target(baseline, degraded, severe_reward_ratio, sharp_value_sens_ratio,
                         order=('embedding', 'comm', 'full')):
    """
    PRIORITY 2: deterministic, rule-based controller (NOT a learned meta-controller) that picks
    an initial repair target from the fingerprint pattern, so target selection stops being a
    manually-typed CLI flag. Rules (thresholds are configurable, not claimed to be optimal):

      1. reward collapsed badly (drop >= severe_reward_ratio of |baseline reward|)
             -> escalate straight to 'full'
      2. value_sensitivity dropped sharply (degraded <= sharp_value_sens_ratio * baseline)
             -> start minimal with 'embedding' (policy/critic still barely see the message;
                a full re-train risks losing everything else the actor learned)
      3. otherwise (the common case: reward down, comm_effect down, value_sensitivity roughly
         held) -> 'comm' (repair what to say and what it means)

    Returns (target, reason) where target is one of `order` and reason is a human-readable
    string for the [CONTROLLER] log line.
    """
    baseline_reward_mag = abs(baseline['reward']) + 1e-6
    reward_drop_ratio = (baseline['reward'] - degraded['reward']) / baseline_reward_mag

    baseline_comm_mag = abs(baseline['comm_effect']) + 1e-6
    comm_drop_ratio = (baseline['comm_effect'] - degraded['comm_effect']) / baseline_comm_mag

    value_sens_ratio = degraded['value_sensitivity'] / max(1e-6, baseline['value_sensitivity'])

    if reward_drop_ratio >= severe_reward_ratio:
        return 'full', (
            "reward collapsed by {:.0%} of baseline magnitude (>= {:.0%} threshold) "
            "-> escalate directly to full".format(reward_drop_ratio, severe_reward_ratio))
    if value_sens_ratio <= sharp_value_sens_ratio:
        return 'embedding', (
            "value_sensitivity dropped sharply ({:.2f}x baseline, <= {:.2f}x threshold) "
            "-> start minimal with embedding".format(value_sens_ratio, sharp_value_sens_ratio))
    return 'comm', (
        "reward down (drop ratio {:.2f}) and comm_effect degraded (drop ratio {:.2f}) "
        "-> standard communication-pathway repair".format(reward_drop_ratio, comm_drop_ratio))


def accept_repair(baseline, degraded, repaired, reward_thresh, comm_thresh, epsilon=1e-6):
    """
    PRIORITY 3: accept/reject rule. Uses normalized recovery of reward AND comm_effect only --
    value_sensitivity and kl stay diagnostic (logged, not gating), because the real experiment
    run showed they do not necessarily move in the same direction as reward/comm_effect during
    repair. Thresholds are configurable and NOT claimed to be scientifically optimal defaults.

    recovery = (repaired - degraded) / (baseline - degraded)

    "Recovery" is only meaningful when the metric actually LOST ground, i.e. when
    (baseline - degraded) > 0. That always holds for reward, because detection requires a
    reward drop. It does NOT always hold for comm_effect: detect_degradation also fires on the
    value_sensitivity branch alone, which permits comm_effect to be flat or even higher under
    degradation. Clamping such a denominator to `epsilon` would divide by 1e-6 and turn a
    0.001 improvement into a 1000.0 "recovery" that clears any threshold automatically -- an
    acceptance manufactured out of numerical noise. Instead, a metric with a non-positive
    denominator reports recovery None and is SKIPPED as a gate: there was nothing to recover,
    so it can neither pass nor fail. If every gate is skipped, the repair is not accepted --
    an acceptance must be positively earned by at least one metric that genuinely dropped.
    """
    def _recovery_ratio(key):
        lost = baseline[key] - degraded[key]
        if lost <= epsilon:
            return None
        return (repaired[key] - degraded[key]) / lost

    reward_recovery = _recovery_ratio('reward')
    comm_recovery = _recovery_ratio('comm_effect')

    gates = []
    skipped = []
    for name, value, thresh in (('reward', reward_recovery, reward_thresh),
                                ('comm_effect', comm_recovery, comm_thresh)):
        if value is None:
            skipped.append(name)
        else:
            gates.append(value >= thresh)

    accepted = bool(gates) and all(gates)
    return accepted, {
        'reward_recovery': reward_recovery,
        'comm_recovery': comm_recovery,
        'skipped_gates': skipped,
    }


def run_causal_adaptive_repair(runner, baseline, degraded, all_args):
    """
    PRIORITY 5: the closed loop. select target -> snapshot -> repair -> measure -> accept/
    reject -> if rejected: rollback + escalate to the next-stronger target -> repeat.

    Escalation only ever moves UPWARD from wherever select_repair_target started (the "least
    invasive valid ordering" per the spec) -- e.g. if the controller starts at 'comm' and it's
    rejected, the next attempt is 'full', never back down to 'embedding'. If every attempt in
    the remaining ladder is rejected, parameters are restored to the pre-repair snapshot and
    the run reports no accepted repair.
    """
    order = ('embedding', 'comm', 'full')
    if all_args.repair_strategy == 'minimum_first':
        initial_target = 'embedding'
        reason = (
            "minimum-first strategy -> start with the least invasive repair "
            "and escalate embedding -> comm -> full only if needed"
        )
        attempt_targets = order
    else:
        initial_target, reason = select_repair_target(
            baseline,
            degraded,
            severe_reward_ratio=all_args.select_severe_reward_ratio,
            sharp_value_sens_ratio=all_args.select_sharp_value_sens_ratio,
            order=order,
        )
        attempt_targets = order[order.index(initial_target):]
    print("\n[CONTROLLER]")
    print("  repair strategy       : {}".format(all_args.repair_strategy))
    print("  selected repair target: {}".format(initial_target))
    print("  reason: {}".format(reason))
    pre_repair_snapshot = runner._snapshot_repairable_state()

    repaired = None
    accepted = False
    accepted_target = None
    escalation_log = []

    for attempt_i, target in enumerate(attempt_targets):
        print("\n[REPAIR ATTEMPT {}/{}] target={}".format(attempt_i + 1, len(attempt_targets), target))
        runner.repair(all_args.repair_iters, target=target)
        repaired = runner.measure_comm_metrics(all_args.measure_episodes, all_args.seed * 100003)
        _print_fingerprint("REPAIRED[{}]".format(target), repaired)

        accepted, acc_info = accept_repair(
            baseline, degraded, repaired,
            reward_thresh=all_args.accept_reward_recovery,
            comm_thresh=all_args.accept_comm_recovery)
        _print_acceptance(acc_info, accepted)

        escalation_log.append({'target': target, 'accepted': accepted, **acc_info})

        if accepted:
            accepted_target = target
            break

        print("  [ROLLBACK] restoring pre-repair parameters")
        runner._restore_repairable_state(pre_repair_snapshot)
        if attempt_i + 1 < len(attempt_targets):
            print("  [ESCALATE] {} -> {}".format(target, attempt_targets[attempt_i + 1]))

    print("\n[FINAL STATUS]")
    if accepted:
        print("  repair accepted: {}".format(accepted_target))
    else:
        print("  all repair attempts rejected")
        print("  original parameters restored")
        print("  (the REPAIRED[...] fingerprint above is diagnostic only -- the loaded")
        print("   policy weights are back to the pre-repair / degraded-environment state)")

    return {
        'mode': 'causal',
        'repair_strategy': all_args.repair_strategy,
        'initial_target': initial_target,
        'initial_reason': reason,
        'accepted': accepted,
        'accepted_target': accepted_target,
        'escalation_log': escalation_log,
    }, repaired, pre_repair_snapshot


def _save_accepted_repair(runner, run_dir, target):
    """
    PRIORITY 9.1: persist an ACCEPTED repair's weights to <run_dir>/models/checkpoint_accepted_<target>/.
    Never called for rejected attempts (those are rolled back, not saved). Without this, an
    accepted repair's weights lived only in the process's memory and were discarded on exit --
    unrecoverable, uninspectable, and unusable for anything past reading the printed numbers.
    """
    tag = "accepted_{}".format(target)
    runner.save(tag=tag)
    saved_path = os.path.join(str(runner.save_dir), "checkpoint_{}".format(tag))
    print("  [SAVE] accepted repair weights written to {}".format(saved_path))
    return saved_path


def _append_results_log(path, row):
    """PRIORITY 9/10: optional one-JSON-line-per-run append, so a future multi-seed/multi-
    perturbation sweep (run externally, e.g. a shell loop over --seed/--mirror_scope/
    --controller) can aggregate results without re-parsing stdout. No sweep is run here."""
    if not path:
        return
    import json
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _log_and_close(all_args, run_dir, baseline, degraded, detect_info, repaired, final_status, envs, eval_envs):
    """PRIORITY 9: bundle every field the spec requires into one results row, PRIORITY 10:
    optionally append it to --results_log, then close the environments."""
    row = {
        'timestamp': time.time(),
        'seed': all_args.seed,
        'model_dir': all_args.model_dir,
        'run_dir': str(run_dir),
        'env_name': all_args.env_name,
        'scenario_name': all_args.scenario_name,
        'mirror_scope': all_args.mirror_scope,
        'perturb_angle': all_args.perturb_angle,
        'controller': all_args.controller,
        'repair_strategy': all_args.repair_strategy,
        'repair_target_arg': all_args.repair_target,
        'repair_iters': all_args.repair_iters,
        # Comparing recovery numbers across different thread counts is invalid -- see the
        # landmine in CODE_WALKTHROUGH.md -- so record both explicitly for later auditing.
        'n_rollout_threads': all_args.n_rollout_threads,
        'n_eval_rollout_threads': all_args.n_eval_rollout_threads,
        'baseline': baseline,
        'degraded': degraded,
        'detect_info': detect_info,
        'repaired': repaired,
        'final_status': final_status,
    }
    _append_results_log(all_args.results_log, row)
    envs.close()
    eval_envs.close()


def _holdout_validate_repair(runner, unmirrored_eval_envs, pre_repair_snapshot, all_args, base_seed):
    """
    TEAM_HANDOFF item 8: re-check an ACCEPTED repair on episodes it was never judged on.

    The accept decision is made on --measure_episodes CRN layouts (seeds base_seed .. +n-1),
    and those same layouts also defined `baseline` and `degraded`. With escalation the
    controller may try up to three targets against that one fixed set, keeping the first to
    clear the thresholds -- so an accepted repair could be one that happens to suit those
    particular layouts rather than a general repair. Their spread is large relative to the
    effect (comm_effect_std ~290 against a mean ~420), so this is a live possibility, not a
    theoretical one.

    This re-derives the whole recovery calculation on a DISJOINT seed block
    (base_seed + --holdout_seed_offset ...), which requires re-measuring all three points on
    the new layouts, not just the repaired one -- `recovery` is relative to baseline and
    degraded, and those are layout-specific too. So the pre-repair weights are temporarily
    restored to measure held-out baseline (normal env) and held-out degraded (mirrored env),
    then the repaired weights are put back to measure held-out repaired.

    Diagnostic only: it never changes the accept decision. It reports whether the decision
    would have survived on evidence that had no part in making it.
    """
    n_ep = all_args.measure_episodes if all_args.holdout_episodes < 0 else all_args.holdout_episodes
    holdout_seed = base_seed + all_args.holdout_seed_offset
    mirrored_eval_envs = runner.eval_envs
    repaired_state = runner._snapshot_repairable_state()

    print("  seeds {} .. {}  (decision used {} .. {} -- disjoint)".format(
        holdout_seed, holdout_seed + n_ep - 1, base_seed, base_seed + all_args.measure_episodes - 1))
    try:
        runner._restore_repairable_state(pre_repair_snapshot)
        runner.eval_envs = unmirrored_eval_envs
        ho_baseline = runner.measure_comm_metrics(n_ep, holdout_seed)
        runner.eval_envs = mirrored_eval_envs
        ho_degraded = runner.measure_comm_metrics(n_ep, holdout_seed)

        runner._restore_repairable_state(repaired_state)
        ho_repaired = runner.measure_comm_metrics(n_ep, holdout_seed)
    finally:
        # The accepted repair must be what is left loaded, whatever happened above.
        runner.eval_envs = mirrored_eval_envs
        runner._restore_repairable_state(repaired_state)

    _print_fingerprint("HELD-OUT BASELINE", ho_baseline)
    _print_fingerprint("HELD-OUT DEGRADED", ho_degraded)
    _print_fingerprint("HELD-OUT REPAIRED", ho_repaired)

    confirms, ho_info = accept_repair(
        ho_baseline, ho_degraded, ho_repaired,
        reward_thresh=all_args.accept_reward_recovery,
        comm_thresh=all_args.accept_comm_recovery)
    _print_acceptance(ho_info, confirms)

    if confirms:
        print("  >>> HELD-OUT CONFIRMS: the repair clears the same thresholds on layouts that")
        print("      played no part in accepting it.")
    else:
        print("  >>> HELD-OUT DOES NOT CONFIRM: the repair cleared the thresholds on the episodes")
        print("      used to accept it, but not on fresh ones. Treat the accepted recovery as")
        print("      specific to those layouts, not as a general repair, until this is resolved.")

    return {
        'holdout_episodes': n_ep,
        'holdout_seed': holdout_seed,
        'holdout_baseline': ho_baseline,
        'holdout_degraded': ho_degraded,
        'holdout_repaired': ho_repaired,
        'holdout_reward_recovery': ho_info['reward_recovery'],
        'holdout_comm_recovery': ho_info['comm_recovery'],
        'holdout_skipped_gates': ho_info['skipped_gates'],
        'holdout_confirms': confirms,
    }


def _measure_normal_env_retention(runner, unmirrored_eval_envs, baseline, all_args, base_seed):
    """
    Re-measure an ACCEPTED repair on the ORIGINAL (unmirrored) environment.

    Without this, the pipeline only ever sees the repaired policy inside the perturbed world,
    so "the repair worked" and "the repair adapted to the mirror by abandoning the original
    task" are indistinguishable -- both produce a good mirrored-env fingerprint. Restoring
    reward on the changed environment while destroying it on the original one is a materially
    weaker result than recovering on both, and the difference should be measured, not assumed.

    Uses the same CRN seeds as the baseline measurement, so this is directly paired with it.
    Diagnostic only: it is reported, never used to gate acceptance.
    """
    mirrored = runner.eval_envs
    try:
        runner.eval_envs = unmirrored_eval_envs
        normal_after = runner.measure_comm_metrics(all_args.measure_episodes, base_seed)
    finally:
        runner.eval_envs = mirrored

    _print_fingerprint("NORMAL-ENV AFTER REPAIR", normal_after)
    reward_delta = normal_after['reward'] - baseline['reward']
    comm_delta = normal_after['comm_effect'] - baseline['comm_effect']
    baseline_mag = abs(baseline['reward']) + 1e-6
    print("  vs. baseline on the SAME layouts: reward {:+.1f} ({:+.1%} of |baseline|), "
          "comm_effect {:+.1f}".format(reward_delta, reward_delta / baseline_mag, comm_delta))
    if reward_delta < -0.10 * baseline_mag:
        print("  [WARNING] the repaired policy is >10% worse than baseline on the ORIGINAL task --")
        print("            recovery in the changed environment came at a real cost to the "
              "unperturbed one.")
    return {
        'normal_after_repair': normal_after,
        'normal_reward_delta_vs_baseline': reward_delta,
        'normal_comm_effect_delta_vs_baseline': comm_delta,
    }


def _print_acceptance(acc_info, accepted):
    """Print the accept/reject block. Recovery is None for any metric that did not lose
    ground under degradation -- there was nothing to recover, so that gate is skipped rather
    than scored (see accept_repair)."""
    def fmt(v):
        return "n/a (nothing lost)" if v is None else "{:.1%}".format(v)
    print("\n[ACCEPTANCE]")
    print("  reward recovery       : {}".format(fmt(acc_info['reward_recovery'])))
    print("  communication recovery: {}".format(fmt(acc_info['comm_recovery'])))
    if acc_info.get('skipped_gates'):
        print("  gates skipped (no degradation to recover): {}".format(
            ", ".join(acc_info['skipped_gates'])))
    print("  decision: {}".format("ACCEPT" if accepted else "REJECT"))


def _print_fingerprint(label, fp):
    print("  [{}] reward={:.1f}  no_msg_reward={:.1f}  comm_effect={:.1f}±{:.1f}  "
          "value_sens={:.3f}  kl={:.3f}".format(
              label, fp['reward'], fp['no_msg_reward'], fp['comm_effect'],
              fp['comm_effect_std'], fp['value_sensitivity'], fp['kl']))


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='simple_spread')
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int, default=2)
    # Phase 2/3 controls
    parser.add_argument('--mirror_scope', type=str, default='partner_full',
                        choices=['all', 'partner', 'partner_full'],
                        help="which observation channels the environment change mirrors "
                             "(see mirror_wrapper.compute_mirror_indices).")
    parser.add_argument('--perturb_angle',type=float,default=None,
                        help=("Optional Task-8 partner-position rotation in degrees. "
                              "0 = no perturbation, 180 = partner_full-equivalent. "
                              "If omitted, the existing --mirror_scope behaviour is used."),)
    parser.add_argument('--measure_episodes', type=int, default=8,
                        help="CRN-paired episodes per condition when measuring a fingerprint.")
    parser.add_argument('--repair_iters', type=int, default=15,
                        help="number of PPO update iterations during online repair.")
    parser.add_argument('--repair_target', type=str, default=None,
                        choices=['embedding', 'comm', 'full', 'noncomm'],
                        help="Default None: the causal controller (select_repair_target) picks "
                             "the target automatically from the fingerprint, with escalation on "
                             "rejection. Pass explicitly to BYPASS the controller and escalation "
                             "and run exactly one fixed-target repair attempt -- this is also how "
                             "the 'noncomm' (Priority 8) non-communication control arm is run, "
                             "since it is a control condition rather than part of the adaptive "
                             "mechanism. 'embedding'/'comm'/'full' can still be forced this way "
                             "for manual ablation.")
    parser.add_argument('--detect_k_sigma', type=float, default=2.0,
                        help="sigma multiplier for the comm-benefit degradation band.")
    parser.add_argument('--detect_min_ratio', type=float, default=0.5,
                        help="degradation if comm_effect falls below this fraction of baseline.")
    parser.add_argument('--detect_reward_drop_ratio', type=float, default=0.15,
                        help="detect_degradation requires reward to have dropped by at least "
                             "this fraction of |baseline reward| in ADDITION to a comm-side "
                             "signal (comm_effect or value_sensitivity) before triggering -- "
                             "reward-down-with-comm-stable, or comm-down-with-reward-stable, "
                             "must not trigger repair on their own.")
    parser.add_argument('--no_repair', action='store_true', default=False,
                        help="only detect; skip the repair stage. Kept as the no-repair "
                             "experimental control regardless of --controller.")
    # TEAM_HANDOFF item 8: held-out re-check of an accepted repair.
    parser.add_argument('--holdout_episodes', type=int, default=-1,
                        help="episodes for the held-out re-check of an ACCEPTED repair. -1 "
                             "(default) means use --measure_episodes; 0 disables the check. "
                             "Runs on a disjoint seed block so the accepted repair is judged "
                             "on layouts that had no part in accepting it. Diagnostic only -- "
                             "it never changes the accept/reject decision.")
    parser.add_argument('--holdout_seed_offset', type=int, default=1000,
                        help="offset added to the base CRN seed for the held-out block. Must "
                             "exceed --measure_episodes so the two blocks cannot overlap.")
    parser.add_argument('--allow_unpaired_eval', action='store_true', default=False,
                        help="bypass the --n_eval_rollout_threads == 1 guard. CRN pairing only "
                             "works in-process (see _seed_eval_envs), so above 1 thread every "
                             "condition starts from a DIFFERENT layout and comm_effect stops "
                             "being a paired quantity. Debugging only -- results produced with "
                             "this flag are not comparable to any CRN-paired run.")

    # PRIORITY 7/10: which trigger/target-selection strategy runs the pipeline.
    parser.add_argument('--controller', type=str, default='causal',
                        choices=['causal', 'reward_only'],
                        help="'causal' (default): detect_degradation (comm-aware trigger) + "
                             "select_repair_target + escalation loop. 'reward_only': PRIORITY 7 "
                             "baseline -- triggers purely off reward drop (detect_degradation_"
                             "reward_only), repairs once with a fixed target (--repair_target if "
                             "given, else 'comm'), no escalation. Exists to test whether causal "
                             "triggering beats the naive alternative; never combine its trigger "
                             "logic with the causal controller's target selection.")
    parser.add_argument('--repair_strategy',type=str,default='threshold',
                        choices=['threshold', 'minimum_first'],
                        help=("Repair ordering for the causal controller. "
                              "'threshold' uses select_repair_target() to choose the initial target. "
                              "'minimum_first' always tries embedding -> comm -> full, "
                              "stopping at the first accepted repair."))
    parser.add_argument('--reward_only_drop_ratio', type=float, default=0.30,
                        help="--controller reward_only trigger threshold: fraction of |baseline "
                             "reward| that must be lost to trigger repair.")

    # PRIORITY 2: select_repair_target thresholds (configurable, not claimed optimal).
    parser.add_argument('--select_severe_reward_ratio', type=float, default=0.50,
                        help="select_repair_target: reward drop ratio (of |baseline reward|) "
                             "at/above which the controller escalates straight to 'full'.")
    parser.add_argument('--select_sharp_value_sens_ratio', type=float, default=0.50,
                        help="select_repair_target: if degraded value_sensitivity falls to at/"
                             "below this fraction of baseline, start with 'embedding'.")

    # PRIORITY 3: accept_repair thresholds (configurable, not claimed optimal).
    parser.add_argument('--accept_reward_recovery', type=float, default=0.30,
                        help="accept_repair: minimum fraction of lost reward that must be "
                             "recovered for a repair attempt to be accepted.")
    parser.add_argument('--accept_comm_recovery', type=float, default=0.30,
                        help="accept_repair: minimum fraction of lost comm_effect that must be "
                             "recovered for a repair attempt to be accepted.")

    # PRIORITY 9/10: optional machine-readable results log (append-only, one JSON line/run).
    parser.add_argument('--results_log', type=str, default=None,
                        help="if set, append one JSON line summarizing this run to this path. "
                             "Intended for an external multi-seed/multi-perturbation sweep to "
                             "aggregate later -- use onpolicy/scripts/aggregate_repair_results.py.")

    # Where this run's logs/models/messages folders (and, if a repair is accepted, the
    # repaired checkpoint) get written. Default avoids littering --model_dir: it writes to
    # a sibling "<checkpoint-dir-name>_repair_runs/<run_id>/" folder instead, one level up.
    parser.add_argument('--run_dir', type=str, default=None,
                        help="output directory for this run. Default: a fresh timestamped "
                             "folder next to --model_dir (never inside it), named "
                             "<checkpoint_dir_name>_repair_runs/<controller>_<target>_seed<N>_<timestamp>/.")
    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # Match train_mpe.py: mappo is non-recurrent.
    if all_args.algorithm_name == "mappo":
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name in ("r_mappo", "macppo"):
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), "check recurrent policy!"

    if all_args.model_dir is None:
        raise ValueError("--model_dir is required (path to a trained actor.pt/critic.pt).")

    # CRN pairing is what makes comm_effect trustworthy at 6-8 episodes: every condition is
    # forced to start from the identical layout. It is implemented by _seed_eval_envs(), which
    # can only reseed the CURRENT process's global NumPy RNG -- the vec envs expose no seed()
    # method to forward through (there is none in onpolicy/envs/env_wrappers.py). Since
    # _make_mpe_vec_env returns a SubprocVecEnv as soon as n_threads > 1, and a parent-process
    # seed does not cross a process boundary, more than one eval thread silently un-pairs every
    # condition: comm_effect becomes a difference of two independently-sampled noisy means and
    # comm_effect_std stops describing paired spread. Nothing downstream would crash or warn,
    # so this is checked up front rather than left to be discovered in the numbers.
    if all_args.n_eval_rollout_threads != 1 and not all_args.allow_unpaired_eval:
        raise ValueError(
            "--n_eval_rollout_threads must be 1 (got {}).\n"
            "CRN-paired measurement only works in-process; above 1 thread the eval envs run in\n"
            "separate processes that the seeding never reaches, so comm_effect / comm_effect_std\n"
            "and every recovery ratio derived from them become unpaired noise -- silently.\n"
            "Re-run with --n_eval_rollout_threads 1, or pass --allow_unpaired_eval if you\n"
            "explicitly want an unpaired (non-comparable) run for debugging."
            .format(all_args.n_eval_rollout_threads))
    # The held-out block must not overlap the decision block, or it is not held out at all.
    n_holdout = (all_args.measure_episodes if all_args.holdout_episodes < 0
                 else all_args.holdout_episodes)
    if n_holdout > 0 and all_args.holdout_seed_offset < all_args.measure_episodes:
        raise ValueError(
            "--holdout_seed_offset ({}) must be >= --measure_episodes ({}), otherwise the "
            "held-out seed block overlaps the block used to accept the repair and the "
            "re-check is not independent.".format(
                all_args.holdout_seed_offset, all_args.measure_episodes))

    if all_args.allow_unpaired_eval and all_args.n_eval_rollout_threads != 1:
        print("\n[WARNING] --allow_unpaired_eval with n_eval_rollout_threads={}: measurements are "
              "NOT CRN-paired.\n          comm_effect and all recovery ratios are not comparable "
              "to paired runs.".format(all_args.n_eval_rollout_threads))

    device = torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)

    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    # Envs: a train env (for repair rollouts) and an eval env (for measurement).
    envs = _make_mpe_vec_env(all_args, all_args.n_rollout_threads, all_args.seed)
    eval_envs = _make_mpe_vec_env(all_args, all_args.n_eval_rollout_threads, all_args.seed * 50000)

    # Default run_dir: a sibling folder next to --model_dir, never inside it (avoids
    # littering the checkpoint directory -- see the run_dir gotcha documented in
    # PROJECT_OVERVIEW.md §6.4). Unique per run so parallel/repeated invocations don't clobber.
    if all_args.run_dir is not None:
        run_dir = Path(all_args.run_dir)
    else:
        model_dir_path = Path(all_args.model_dir)
        if all_args.repair_target is not None:
            repair_label = all_args.repair_target
        else:
            repair_label = all_args.repair_strategy

        run_id = "{}_{}_seed{}_{}".format(
            all_args.controller,
            repair_label,
            all_args.seed,
            time.strftime("%Y%m%d_%H%M%S"),
        )
        run_dir = (model_dir_path.parent / (model_dir_path.name + "_repair_runs") / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
    }
    # Force the non-wandb branch in base_runner so save_dir/run_dir are plain paths.
    all_args.use_wandb = False

    runner = RepairRunner(config)  # restores from model_dir inside __init__

    mirror_idx = compute_mirror_indices(
        all_args.num_agents, all_args.num_landmarks, scope=all_args.mirror_scope)
    print("\n=== Phase 2/3 pipeline ===")
    print("seed            :", all_args.seed)
    print("model_dir       :", all_args.model_dir)
    print("run_dir         :", str(run_dir))
    print("env / scenario  :", all_args.env_name, "/", all_args.scenario_name)
    print("mirror scope    :", all_args.mirror_scope, "-> indices", mirror_idx)
    if all_args.perturb_angle is not None:
        print("Task-8 partner rotation :",all_args.perturb_angle,"degrees",)
    print("controller      :", all_args.controller)
    print("repair strategy :", all_args.repair_strategy)
    print("repair target   :", all_args.repair_target if all_args.repair_target is not None
          else "(auto-selected by controller)")
    # n_rollout_threads controls how much experience repair() collects per PPO iteration --
    # comparing repair results across runs with different values is comparing apples to
    # oranges (see the landmine in CODE_WALKTHROUGH.md and the caveat in PROJECT_OVERVIEW.md
    # §5.9). Printed explicitly so it's never silently forgotten when comparing runs later.
    print("n_rollout_threads (repair rollout) :", all_args.n_rollout_threads)
    print("n_eval_rollout_threads (fingerprint):", all_args.n_eval_rollout_threads)

    # PRIORITY 1B: with exactly 2 agents, attention-based aggregation is mathematically inert --
    # see _apply_attention_aggregation in rMAPPOPolicy.py: softmax over the single unmasked
    # sender is always 1.0 regardless of attention_weight's value, so its gradient is also
    # exactly zero here. Logged explicitly so 'comm'/'full' repair are never described as
    # repairing "who to trust" in this configuration.
    if runner.num_agents == 2:
        print("\n[NOTE] num_agents=2: attention-based message aggregation reduces to selecting")
        print("       the sole available sender unconditionally (softmax over one unmasked")
        print("       class is always 1.0). attention_weight receives ZERO gradient here and")
        print("       does not learn 'who to trust', even though 'comm'/'full' repair targets")
        print("       still mark it trainable (kept for forward-compatibility with >2 agents).")

    base_seed = all_args.seed * 100003

    # 1) BASELINE on the normal environment.
    print("\n[1] Baseline fingerprint (normal environment):")
    baseline = runner.measure_comm_metrics(all_args.measure_episodes, base_seed)
    _print_fingerprint("BASELINE", baseline)

    # 2) Apply the environment change (one-sided mirror) and measure degradation.
    runner.eval_envs = _wrap_perturbed_env(
    eval_envs,
    all_args,
    mirror_idx,
    )   
    print("\n[2] Degraded fingerprint (mirrored environment):")
    degraded = runner.measure_comm_metrics(all_args.measure_episodes, base_seed)
    _print_fingerprint("DEGRADED", degraded)

    # 3) Detection.
    if all_args.controller == 'reward_only':
        is_degraded, detect_info = detect_degradation_reward_only(
            baseline, degraded, all_args.reward_only_drop_ratio)
        print("\n[3] Detection (PRIORITY 7 REWARD-ONLY BASELINE"
              " -- comm_effect/value_sensitivity/kl NOT used):")
        print("  reward drop ratio : {:.2f}  (threshold {:.2f})".format(
            detect_info['reward_drop_ratio'], detect_info['reward_drop_ratio_threshold']))
        print("  >>> ENVIRONMENT CHANGE {} (reward-only trigger)".format(
            "DETECTED" if is_degraded else "NOT detected"))
    else:
        is_degraded, detect_info = detect_degradation(
            baseline, degraded, all_args.detect_k_sigma, all_args.detect_min_ratio,
            all_args.detect_reward_drop_ratio)
        print("\n[3] Detection (causal: reward-degraded AND comm-related, both required):")
        print("  reward drop ratio     : {:.2f}  (threshold {:.2f}, degraded={})".format(
            detect_info['reward_drop_ratio'], detect_info['reward_drop_ratio_threshold'],
            detect_info['reward_degraded']))
        print("  comm_effect threshold : {:.1f}   (baseline {:.1f} -> degraded {:.1f}, degraded={})".format(
            detect_info['comm_threshold'], baseline['comm_effect'], degraded['comm_effect'],
            detect_info['comm_degraded']))
        print("  value-sensitivity     : {:.3f} -> {:.3f}  (degraded={})".format(
            baseline['value_sensitivity'], degraded['value_sensitivity'], detect_info['value_degraded']))
        print("  >>> ENVIRONMENT CHANGE {}".format("DETECTED" if is_degraded else "NOT detected"))

    if not is_degraded or all_args.no_repair:
        print("\nDone (no repair stage).")
        final_status = {'mode': 'no_repair_control' if all_args.no_repair else 'not_degraded'}
        _log_and_close(all_args, run_dir, baseline, degraded, detect_info, None, final_status, envs, eval_envs)
        return

    # 4) Online repair on the mirrored environment.
    runner.envs = _wrap_perturbed_env(
    envs,
    all_args,
    mirror_idx,
    )

    if all_args.repair_target is not None:
        # PRIORITY 8 / manual override: exactly one fixed-target attempt, no controller,
        # no escalation. This is also how the 'noncomm' control arm is invoked.
        print("\n[4] Manual repair ({} iters, target={}) -- controller/escalation bypassed:".format(
            all_args.repair_iters, all_args.repair_target))
        snapshot = runner._snapshot_repairable_state()
        runner.repair(all_args.repair_iters, target=all_args.repair_target)
        repaired = runner.measure_comm_metrics(all_args.measure_episodes, base_seed)
        _print_fingerprint("REPAIRED", repaired)
        accepted, acc_info = accept_repair(
            baseline, degraded, repaired,
            all_args.accept_reward_recovery, all_args.accept_comm_recovery)
        _print_acceptance(acc_info, accepted)
        saved_path = None
        if not accepted:
            print("  [ROLLBACK] restoring pre-repair parameters (manual mode: no escalation)")
            runner._restore_repairable_state(snapshot)
        else:
            saved_path = _save_accepted_repair(runner, run_dir, all_args.repair_target)
        print("\n[FINAL STATUS]")
        print("  repair {}: {}".format("accepted" if accepted else "REJECTED (rolled back)",
                                        all_args.repair_target))
        final_status = {'mode': 'manual', 'target': all_args.repair_target,
                        'accepted': accepted, 'saved_path': saved_path, **acc_info}

    elif all_args.controller == 'reward_only':
        # PRIORITY 7: fixed target, single attempt, no escalation -- isolates the TRIGGER
        # decision as the only difference from the causal controller. Fixed at 'comm' to stay
        # comparable with the causal controller's most common default target.
        target = 'comm'
        print("\n[4] Reward-only-triggered repair ({} iters, target={}, fixed, no escalation):".format(
            all_args.repair_iters, target))
        snapshot = runner._snapshot_repairable_state()
        runner.repair(all_args.repair_iters, target=target)
        repaired = runner.measure_comm_metrics(all_args.measure_episodes, base_seed)
        _print_fingerprint("REPAIRED", repaired)
        accepted, acc_info = accept_repair(
            baseline, degraded, repaired,
            all_args.accept_reward_recovery, all_args.accept_comm_recovery)
        _print_acceptance(acc_info, accepted)
        saved_path = None
        if not accepted:
            print("  [ROLLBACK] restoring pre-repair parameters (reward_only baseline: no escalation)")
            runner._restore_repairable_state(snapshot)
        else:
            saved_path = _save_accepted_repair(runner, run_dir, target)
        print("\n[FINAL STATUS]")
        print("  repair {}: {}".format("accepted" if accepted else "REJECTED (rolled back)", target))
        final_status = {'mode': 'reward_only', 'target': target, 'accepted': accepted,
                        'saved_path': saved_path, **acc_info}

    else:
        # PRIORITY 2/3/4/5: the causal adaptive controller -- auto-select, snapshot, repair,
        # accept/reject, escalate on rejection.
        final_status, repaired, snapshot = run_causal_adaptive_repair(
            runner, baseline, degraded, all_args)
        saved_path = None
        if final_status.get('accepted'):
            saved_path = _save_accepted_repair(runner, run_dir, final_status['accepted_target'])
        final_status['saved_path'] = saved_path

    # 5) Retention check: an accepted repair is re-measured on the ORIGINAL environment, so
    # "repaired communication" can be told apart from "adapted to the mirror at the cost of the
    # original task". Skipped when nothing was accepted -- a rejected attempt has already been
    # rolled back, so the policy is bit-identical to the one that produced `baseline`.
    if final_status.get('accepted'):
        print("\n[5] Retention check on the ORIGINAL (unmirrored) environment:")
        final_status.update(
            _measure_normal_env_retention(runner, eval_envs, baseline, all_args, base_seed))

    # 6) Held-out re-check (TEAM_HANDOFF item 8): re-derive the recovery on a disjoint block of
    # layouts that had no part in the accept decision. `snapshot` holds the pre-repair weights,
    # needed because held-out baseline/degraded must be re-measured on the new layouts too.
    if final_status.get('accepted') and all_args.holdout_episodes != 0:
        print("\n[6] Held-out re-check on episodes the decision never saw:")
        final_status.update(
            _holdout_validate_repair(runner, eval_envs, snapshot, all_args, base_seed))

    print("\n=== Summary ===")
    print("                  reward    comm_effect   value_sens      kl")
    print("  baseline    : {:>8.1f}   {:>10.1f}   {:>8.3f}  {:>6.3f}".format(
        baseline['reward'], baseline['comm_effect'], baseline['value_sensitivity'], baseline['kl']))
    print("  degraded    : {:>8.1f}   {:>10.1f}   {:>8.3f}  {:>6.3f}".format(
        degraded['reward'], degraded['comm_effect'], degraded['value_sensitivity'], degraded['kl']))
    if repaired is not None:
        print("  repaired    : {:>8.1f}   {:>10.1f}   {:>8.3f}  {:>6.3f}".format(
            repaired['reward'], repaired['comm_effect'], repaired['value_sensitivity'], repaired['kl']))
        recov = repaired['comm_effect'] - degraded['comm_effect']
        print("  comm_effect recovered by {:.1f} ({:.0f}% of the lost benefit)".format(
            recov, 100.0 * recov / max(1e-6, baseline['comm_effect'] - degraded['comm_effect'])))

    _log_and_close(all_args, run_dir, baseline, degraded, detect_info, repaired, final_status, envs, eval_envs)


if __name__ == "__main__":
    main(sys.argv[1:])
