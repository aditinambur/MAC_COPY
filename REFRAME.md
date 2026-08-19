# Reframe — What Changed From the Original Plan

The original plan described three layers, each with a list of challenges to solve. Some of those
challenges turned out to be real. Others never came up, because the system we built works
differently than the plan assumed. And a different set of problems appeared that nobody predicted.

This document explains the difference, so the write-up describes what we actually did.

---

## The one change that explains most of the others

**The plan assumed a system that runs continuously while the agents train.**
Messages would drift, the system would notice, repair on the fly, and keep going.

**What we built runs once, offline.** Load a trained agent → measure how well it communicates →
break the environment → check whether communication broke → repair once → keep the repair or undo
it. Then stop.

That single difference removes four of the seven planned challenges, because they were all about
things going wrong *over time* — agents drifting out of sync, repairs firing repeatedly, the
system oscillating. None of that can happen if it only runs once.

**This is worth stating plainly in the paper.** Those challenges were avoided by design, not
solved. Claiming otherwise is the kind of thing a reviewer checks.

---

## Layer 1 — Measuring whether communication matters

### What we built

We take the trained agent and ask it to act twice on exactly the same situation: once with the
message it actually received, once with that message blanked out. The difference tells us what the
message caused. We record two things — how much the agent's *behaviour* changed, and how much its
*expected reward* changed. Separately we measure whether the messages actually help the team score
better, which is the number that matters most.

We did **not** build a model of the other agent, an information bottleneck, or a causal
regulariser. Those were in the plan and were not needed.

### Planned: "the causal signal will be noisy"

**This was right, and it was the hardest problem.** The measurement swings by ±290–440 while the
effect itself is only around +300–535. The noise is nearly as big as the thing being measured.

**We solved it differently than planned.** Instead of smoothing and confidence thresholds, we
force every comparison to start from *identical* random situations. Then layout luck cancels out
and only the message effect remains. This is why 6 episodes is enough — unpaired, it would take
far more.

We also re-check every accepted repair on a completely fresh set of episodes it was never judged
on. That has already caught two repairs that looked successful and were not.

### Planned: "it depends on the quality of the critic"

**Half of this applies.** We never built the model-of-other-agents, so that half is moot.

But the critic dependency caused a real problem, just not the predicted one. The value-based
measure turned out to be so unstable that it broke our checkpoint selection twice — once picking
an agent from 3.5% into training because of a single freak reading. We now treat that measure as
*informational only*: it gets reported but decides nothing.

### Planned: "balancing multiple training objectives"

**Never came up.** There are no competing objectives — training is ordinary PPO.

The one real design decision was cutting the value function's ability to shape the messages. This
matters: without that cut, "the messages are useful" could just mean "we trained the messages to
make the value function's job easier." With it, the claim is about behaviour. We checked this did
not weaken communication — the strongest agent we trained came from the current code.

---

## Layer 2 — Repairing communication

### What we built

Retrain a small, selected slice of the network on the broken environment — anywhere from 320
parameters (just the word meanings) to 10,192 (the whole action network). Then check whether it
worked, and if not, undo it completely and try a bigger slice.

### Planned: "agents will get out of sync during repair"

**Never came up.** Both agents share one network, so they update together. There is no partner
left holding the old meaning of a word.

*(This would be a real problem if each agent had its own network. Worth mentioning as future work,
not as something we solved.)*

### Planned: "repairing too often will cause oscillation"

**Never came up.** Repair runs once. Nothing to oscillate.

### What actually went wrong instead: accepting a bad repair

The real risk was that a repair *looks* successful on the episodes used to judge it, and isn't.

Two things address it. First, a rejected repair is undone completely — weights, both optimisers,
and the reward-scaling statistics, all restored exactly. Second, every accepted repair is
re-tested on fresh episodes. **Two repairs passed their own test and then failed the fresh one.**
Without that check we would have written both up as successes.

---

## Layer 3 — Deciding when to repair

### What we built

A simple rule: only repair if performance dropped **and** communication looks like the cause.
Both must be true.

### Planned: "telling real drift apart from random noise"

**This was right, and the two-condition rule is what solves it.** If communication looks worse but
performance is fine, there is nothing worth fixing. If performance drops but communication is
intact, it is not a communication problem.

**We have direct evidence it works.** Two separately trained agents had clearly degraded
communication while their scores barely moved. The system declined to repair both. We then forced
it to repair them anyway — and in one case the fix failed the fresh-episode check, in the other
there was no valid fix at all. **The system was right to refuse.**

### Planned: "the trigger will be unstable and flip back and forth"

**Never came up.** It runs once.

### Planned: a *learned* controller

**Not built.** What we have is a hand-written rule, which the original plan explicitly called for
as a first step. That is fine to report as a stand-in — but note that **it has chosen the same
option every single time it has run.** One of its three branches has never activated. So we can
say the system escalates correctly when a repair fails; we cannot yet say it *chooses* between
options, because it has never been given a situation where it would choose differently.

---

## What actually took the effort (none of it was in the plan)

1. **The "best agent" picker is broken.** It failed in two of three training runs, each time
   burying the best agent that run produced. One pick came from 3.5% into training.

2. **Training is unreliable.** Roughly one run in three produces an agent too weak to test on at
   all — its messages are barely worth anything, so there is nothing to break or repair.

3. **We were grading on the same test cases used to pick the answer.** The same six episodes
   defined the baseline *and* judged up to three repair attempts. Fixed by re-testing on fresh
   episodes — which then rejected repairs the original test had accepted.

4. **Repair costs something.** After a repair the agents are worse at the *original*, unbroken
   task — by up to 17% in the worst case, and more the bigger the repair. This is a real tradeoff
   to report, not a flaw to hide.

5. **Several decisions came down to rounding.** Two of our three results hinged on a number
   landing within 0.002 of a cutoff. They went our way, but that is luck, and it needs saying.

---

## Summary

| | planned | what happened |
|---|---|---|
| **Measuring communication** | noisy signals | ✅ real — solved by paired comparisons, not smoothing |
| | critic quality | ⚠️ partly — the value measure is unreliable, so it now decides nothing |
| | balancing objectives | ❌ never came up |
| **Repairing** | agents out of sync | ❌ never came up — shared network |
| | oscillation | ❌ never came up — runs once |
| | *(actual)* bad repairs slipping through | ✅ solved by full undo + fresh-episode re-testing |
| **Deciding when** | noise vs real drift | ✅ real — solved by requiring both conditions |
| | unstable trigger | ❌ never came up — runs once |
| | learned controller | ❌ not built, and the hand-written one has never picked differently |

**The honest framing for the paper:** the measurement and detection layers work and have been
tested on real data. The repair works and survives re-testing on fresh episodes. The
decision-making layer is a simple rule that has not yet been shown to make a real choice. And
several of the difficulties in the original plan do not apply to a system that runs once — they
were designed around, not overcome.
