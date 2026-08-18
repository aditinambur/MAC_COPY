# Project Handoff — What's Done, What's Next

## What this project actually is, in plain terms

Two AI agents are learning to play a cooperative game together (spread out and cover some
landmarks), and while doing it, they also learn to send each other short messages to help
coordinate. On top of that, we built a system that:

1. **Checks if the messages are actually helping** — not just assumes they are, but
   actually tests it by taking a message away and seeing what changes.
2. **Deliberately breaks something** — we flip part of what one agent can see about the
   other agent's position, on purpose, to simulate "something in the environment changed."
3. **Notices the break happened** — checks if performance got worse *and* if it's
   specifically the messages that stopped helping (not just a random bad day).
4. **Automatically tries to fix it** — retrains a small part of the message system using
   fresh experience in the broken environment.
5. **Checks if the fix actually worked**, and either keeps it or throws it away and tries
   a bigger fix.

That whole loop — break it, notice it, fix it, check the fix — is built and has been run
for real, not just on paper.

## What's already been built and confirmed working

- The agents genuinely learn to communicate — not fake or symbolic, the messages actually
  change their behavior and their expected reward.
- We can measure, honestly, whether a message actually helped or not (not just whether
  the agents happened to do well).
- We have a reliable way to deliberately break communication without breaking the whole
  game.
- The system correctly notices when communication broke, and correctly ignores cases
  where it *didn't* break (so it doesn't fix things that aren't broken).
- The system picks a fix on its own, tries it, checks if it worked, and if not, undoes it
  and tries something bigger.
- We ran this for real, multiple times, and it worked as intended — including one time
  where the first fix attempt failed and the system correctly noticed, undid it, and
  tried a stronger fix that worked.
- We ran the two comparison checks below once each. Both are single test runs, not
  confirmed patterns yet — that's exactly what the required list below fixes.
- **Accepted fixes are automatically re-checked on test runs they were never judged on.**
  The system grades a fix on 6 randomly generated episodes — but those same 6 also defined
  the "before" and "broken" numbers, and the system gets up to 3 tries against them. So
  whenever a fix is accepted, the pipeline now re-measures all three points on a completely
  separate batch of episodes and re-runs the same accept test, printing
  `HELD-OUT CONFIRMS` or `HELD-OUT DOES NOT CONFIRM`. **This has been run and it confirmed**
  (see `PROJECT_OVERVIEW.md` §5.11). Nothing to do here — just don't ignore the verdict when
  you run items 2-5: a `DOES NOT CONFIRM` means that run's recovery number does not get
  written up as a success.

## What we already found (three runs each — consistent, but not yet statistically proven)

- **The "smart" system beat the "naive" one — twice, but only just.** The naive version
  decides whether to fix things using only the score. On two different trained agents, the
  smart version spotted the problem and produced a fix that held up on fresh test runs, while
  the naive version did nothing at all. The catch: both times the score drop landed a hair
  under the naive version's cutoff (0.2995 and 0.2981 against a cutoff of 0.30). So the two
  genuinely disagree, and the smart one was right both times, but the margin was tiny. Item 2
  below still needs a case where they disagree by a comfortable margin.
- **Fixing communication specifically clearly beat a generic fix of the same size — twice.**
  On one agent the generic fix looked fine at first and then failed the fresh-test-run check
  outright. On the other it was rejected immediately. Meanwhile the communication fix passed
  both times. This is the strongest evidence we have that the recovery really is about
  repairing communication, though it is still only two runs.
- **The fix costs something on the original task, and we under-called this earlier.** After a
  repair the agents are worse at the *unbroken* version of the game — by 17.3% in the worst
  case so far. The bigger the fix, the bigger the cost. Any write-up should say so.

## Where things stand right now

Everything below was run on the **current** code. Earlier runs on the old communication code
have been retired and are not counted anywhere.

| # | what it is | status |
|---|---|---|
| **1** | **replace the broken "best agent" picker — do this first** | 🔴 **needs a proper fix; everything else depends on it** |
| 2 | run the system on several different trained agents | 🟡 3 agents repaired successfully, but from only **2 training runs** of the 8 needed |
| 3 | compare against the "naive trigger" | 🟡 **3 for 3** — and one of them by a comfortable margin |
| 4 | compare against the "generic fix" | 🟡 **3 for 3** — the real fix won every time |
| 5 | compare against doing nothing | 🔴 needs rethinking, not repeating — it is trivially 0% by construction |
| 6 | check the differences are real, with statistics | 🔴 not started — needs ≥5 agents to reach p<0.05 (item 8 supplies them) |
| 7 | check the cutoff numbers are sensible | 🟡 plenty of evidence gathered, no analysis done |
| 8 | more agents, from genuinely new training runs | 🔴 **8 usable agents needed, from 8 separate runs** (have 2 runs) |
| 9 | test the part that picks how big a fix to try | 🟡 it escalates correctly, but has never once picked anything other than the middle option |
| 10 | break the system a *different* way | 🔴 not started — every result so far uses one kind of break |
| 11 | try it with 3 agents instead of 2 | 🔴 not started — needed before claiming anything about "who to listen to" |

**The short version:** the two comparisons that carry the project (items 3 and 4) have gone
3-for-3 in the right direction. Nothing is proven yet because 3 runs cannot support statistics,
and they come from only 2 training runs. Item 1 is half a day and unblocks everything; items 6 and 8 are what stand between this
and a defensible result.

## What's required next, in priority order

These eleven are the minimum needed before any of the results above can be trusted or
written up. Everything else discussed (different kinds of environment breaks, per-agent
breakdowns, etc.) comes after these, not alongside them.

**1. FIRST — design a replacement for the broken "best agent" picker.**

*Do this before any of the experiments below.* It is about half a day of work, and
skipping it means doing eight manual checkpoint sweeps by hand and risking that someone
runs a full set of experiments on a weak agent without noticing. Everything from item 2
onward depends on picking the right agent to test.

At the end of training the system saves what it thinks is the best version of the agent into a
folder called `checkpoint_best`. **It has now picked badly in two of three runs**, and both times
it buried the best agent that run actually produced:

- One run chose a version from **3.5% into training** — an agent that had barely learned
  anything. A single early evaluation happened to score a freak high value on one of the three
  measures, and because that measure is weighted 100× in the scoring formula, nothing across the
  remaining 1.9 million training steps ever beat it.
- Another chose an agent whose communication was worth **+139**, while the final version of that
  same agent was worth **+535** — nearly four times better, and the best agent produced anywhere
  in this project. That run was written off as a failure for most of a day on the strength of the
  wrong checkpoint.

This is not bad luck. The score takes a running maximum across 60–90 noisy evaluations, so one
early fluke wins permanently and is never revisited.

**This item is not "check it by hand" — that is only the stopgap.** For now, sweep the 3–4
checkpoints with the highest `causal_influence_kl_mean` from that run's `causal_influence.csv`,
run each with `--no_repair`, and pick using the thresholds in item 8. Budget ~4 minutes per
candidate.

**What is actually needed is a better rule.** Some options worth trying and comparing against the
hand-picked answer on the three runs we already have (where we know which checkpoint is right):

- **Average over a window** instead of taking a single evaluation — e.g. score each checkpoint on
  the mean of its own and its neighbours' evals, so one freak reading cannot win.
- **Drop the 100× weighting on value-sensitivity**, or cap its contribution. It is the term that
  caused both failures, and it is the least reliable of the three measures.
- **Score on `comm_effect` directly** rather than a weighted blend — it is the quantity the repair
  experiments actually care about, and it is already measured at every evaluation.
- **Ignore the first ~20% of training** when selecting, since nothing that early is a serious
  candidate.

Whichever is chosen, validate it the same way: does it pick the checkpoint we would have picked
by hand, on all three existing runs? Until something passes that test, do not point
`--model_dir` at `checkpoint_best`.

**2. Run the automatic system multiple times with different starting points (seeds), on
the setup we already have.**
Right now it's only been proven to work once. Each run uses a different random starting
layout and a different stream of randomness during training, so running it several times
checks whether the system reliably notices the problem, picks a sensible fix, and
succeeds — across different conditions — rather than us having gotten lucky on the one
run we tried.

**3. Run the "naive trigger" comparison the same number of times, on the same starting
points.**
This is the project's main claim: that being selective about *when* to fix
communication — only stepping in when communication itself looks like the problem — is
better than a blunt version that fixes things whenever the score drops, for any reason.
Right now the smart version and the naive version made the exact same call on the one test
we ran, so there's currently zero evidence this claim is true. It might be — we just
haven't tested a case where the two versions would actually disagree yet.

**4. Run the "generic fix" comparison the same number of times, on the same starting
points.**
The system's normal fix only retrains the small part of the network responsible for
communication — what an agent says and what its messages mean. The comparison version
retrains a different, similarly small part of the network that has nothing to do with
communication, as a fairness check. If the generic fix works just as well, that would mean
the recovery isn't really about "fixing communication" — it would just show that
retraining *any* similar-sized piece helps. On the one test so far, the real fix did
slightly better, but not by enough to be sure it's a real difference and not luck.

**5. Run the "no fix at all" comparison the same number of times, on the same starting
points.**
This checks what happens if nothing is done after the environment breaks, so we have a
solid "before" number to measure every fix against — confirming that number is stable and
not just noise from one lucky or unlucky run.

**6. Put all of those results together and check if the differences are real, not just
luck.**
With only one run per comparison, we can't tell a genuine pattern from random chance. This
step takes the repeated runs from items 2-5 and actually checks — using proper
statistics — whether "fixing communication beats a generic fix" or "being selective beats
the naive version" are real effects, or just how those particular single runs happened to
go. Without this step, none of the repeated running matters, no matter how many times it's
done.

**7. Check whether the cutoff numbers used throughout the system are actually reasonable,
not just convenient guesses.** *(elevated — do this right after item 6, not last)*
There are eight different cutoff numbers scattered through the system — for example, "the
score has to drop by at least 15% before we suspect a problem," or "at least 30% of the
loss has to be recovered before a fix counts as accepted." Every one of them was picked as
a round, reasonable-sounding number, never tested. On the one real run we have, the score
drop landed at 47% — just 3 points from the 50% cutoff that decides whether the system
tries a small fix first or jumps straight to the biggest one. A few points either way would
have told a completely different story. Once items 2-5 produce several repeated runs, this
step goes back through those cutoff numbers and checks whether the results would have come
out differently with slightly different cutoffs — telling us whether the current numbers
are actually reasonable or just happened to work for the handful of runs tried so far. This
can't run before items 2-5 exist (it needs their output to check against), but it matters
enough that it shouldn't be left until last either — a wrong cutoff quietly affects every
other result on this list, so it's worth confirming early.

**8. Train 8 usable agents, from 8 separate training runs. Budget for about 12 runs.**

Everything so far comes from **two** training runs. For AAMAS the target is **8 usable agents,
each from its own independent training run.** Five is the absolute floor; eight is what makes the
result comfortable.

**Why 5 is the floor.** Both comparisons produce a yes/no outcome per agent, so the natural
analysis is a sign test. Five-for-five gives p = 0.031, six-for-six gives p = 0.016. **Below five
you cannot reach p < 0.05 no matter how clean the results are.** The current three-for-three sits
at p = 0.125 — consistent, but not significant.

**Why 8 rather than 5.** Two reasons specific to this setup:

- **Different checkpoints from the same training run do not count as separate agents.** Two of
  our three current agents come from the same run, 64,000 steps apart. A reviewer will count that
  as one sample, not two. **The unit is the training run.**
- **The naive-trigger comparison does not produce a result on every agent.** It only tells you
  anything when the reward drop lands between 0.15 and 0.30 — above that both versions react and
  you get an uninformative tie, below that neither does. All three of ours happened to land in
  that window, which was luck. At roughly a 50–60% hit rate, 8 agents yields about 5 usable
  trigger comparisons, which just clears the floor. The generic-fix comparison uses every agent,
  so 8 is comfortable there.

**Why 12 training runs.** Roughly one run in three produces an agent too weak to use, so expect
to discard about four.

**What counts as too weak to use.** Run `--no_repair` on a candidate and read two numbers:

- **`comm_effect`** — how much reward the messages are worth. Below about **+150** the agent
  barely uses them. Two of our checkpoints measured +116.9 and +139.1 and were useless here.
- **reward drop under the break** — below about **0.16** the repair has always failed, and below
  0.15 the detector correctly refuses to fire at all.

**What a good agent looks like:** `comm_effect` around **+300 or more** and a reward drop around
**0.24 or more**. All three agents that repaired successfully sat in that range (+326.0 / 0.300,
+534.9 / 0.298, +402.5 / 0.241).

**Time per usable agent:** ~40 min training, ~16 min sweeping its checkpoints (item 1), ~35 min
for the three comparison arms. Call it 90 minutes, mostly unattended. Twelve runs including the
discards is roughly 15 hours of compute — comfortable across two weeks if it is started early and
run in the background rather than one-at-a-time by hand.

### How to report it — two things reviewers check

**Report every agent, including the weak ones you threw away.** "12 training runs, 8 usable,
4 discarded for `comm_effect` below +150" is a *strength* — it shows you know how variable the
training is, and it is honest about selection. Quietly dropping the failures is the single most
common way a paper like this gets rejected. Keep a table of all 12 with their `comm_effect` and
reward drop, and say which ones were used and why.

**Report the trigger comparison as "k of n", on the agents where it actually applied — and list
the drop ratios.** Do not report "the smart trigger beat the naive one" as a flat claim. Say how
many agents produced a usable comparison, how many went each way, and what the margin was in
each case. This matters because two of our three current margins are **0.0005** and **0.0019** —
the smart version was right, but only just. If that pattern repeats across 8 agents, you want it
visible in your own table rather than discovered by a reviewer. The third margin was 0.0594,
which is the kind you can lean on.

**9. Actually test the part of the system that decides how big a fix to try, instead of
letting it default to the same choice every time.**
There's a piece of the system whose whole job is to look at how bad things are and decide
between a small fix, a medium fix, or a big one. So far, every real test has used the exact
same break at the exact same strength — so this piece has picked the same option almost
every single time. Right now there's no real evidence it's doing anything more than always
defaulting to the same answer. It's a hand-written set of rules, not something the system
learned, and it can only prove itself once it's actually given a situation that's clearly
milder or clearly worse than what's been tried so far — which means it depends on testing a
range of break severities, not just repeating the same one. This is related to item 6 (are
the cutoff numbers reasonable) but is a different question: even with perfectly-chosen
cutoff numbers, this piece of the system can't demonstrate it's needed until it's actually
been given a chance to choose differently. Placed last because, unlike items 2-8, it needs
genuinely new experiments (a different break strength), not just repeats of what exists.

**10. Break the system in a second, different way — not just harder or softer.**

Every result in this project comes from **one** kind of break: flipping the agent's sense of
where its partner is. That is enough to show the repair works, but it is not enough to claim the
*detector* is general. As written, "our system notices when communication breaks" really means
"our system notices when this one specific thing breaks."

Item 9 asks for the same break at different strengths. This item is different: a break of a
different **kind**. Two options, both straightforward to add:

- **Agent dropout** — one agent stops sending messages entirely for part of an episode. Tests
  whether the detector notices communication *disappearing* rather than becoming *misleading*.
- **Vocabulary corruption** — shuffle or randomise the token embeddings, so the words still
  arrive but no longer mean what they meant. This is arguably the more interesting one, because
  it is the failure the `embedding` repair target was designed for, and that target has never
  once been used.

Run the same three comparisons on it. If the detector fires and the repair works on a second kind
of break, "perturbation-agnostic" becomes a claim you can defend. If it does not, that is a real
limitation and much better found now.

**11. Try it with 3 agents instead of 2 — before claiming anything about "who to listen to".**

The system contains a learned `attention_weight` whose stated purpose is letting each agent decide
which sender to weight more heavily. **With exactly 2 agents it does nothing at all**, and this is
not a suspicion — it is arithmetic. Each agent has exactly one possible sender (itself is masked
out), and a softmax over a single option is always 1.0 no matter what the weight says. We verified
it three ways: the output is bit-identical across wildly different weight values, the gradient is
exactly zero, and the parameter did not move by a single bit across 2 million training steps.

**So do not describe the system as learning who to trust, or as using attention, while it runs
with 2 agents.** It is scaffolding that currently does nothing.

Running with `--num_agents 3` makes it real: three agents means two possible senders, so the
softmax has an actual choice to make and the parameter starts receiving gradient. That is what
turns the attention mechanism from an unused component into a contribution.

Expect this to need its own training runs and its own tuning — three agents is a harder
coordination problem, and none of the thresholds have been checked at that scale. Placed last
because it is the largest piece of new work on this list, but it is the one that unlocks a claim
the project currently cannot make.
