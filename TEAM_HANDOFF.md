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
  you run items 1-4: a `DOES NOT CONFIRM` means that run's recovery number does not get
  written up as a success.

## What we already found (one test run each — not yet proven, needs repeating)

- **The "smart" system and a "naive" version tied exactly.** The naive version decides
  whether to fix things using only the score, ignoring anything about whether
  communication itself looks broken. On the one test we ran, both versions made the
  identical decision and produced the identical fix. That's not evidence either way yet —
  it just means this particular test wasn't hard enough to tell them apart.
- **Fixing communication specifically did slightly better than a generic fix of the same
  size**, but only by a small margin (about 5-9 percentage points), which could easily be
  random noise from a single run rather than a real effect.

## What's required next, in priority order

These eight are the minimum needed before any of the results above can be trusted or
written up. Everything else discussed (different kinds of environment breaks, per-agent
breakdowns, etc.) comes after these, not alongside them.

**1. Run the automatic system multiple times with different starting points (seeds), on
the setup we already have.**
Right now it's only been proven to work once. Each run uses a different random starting
layout and a different stream of randomness during training, so running it several times
checks whether the system reliably notices the problem, picks a sensible fix, and
succeeds — across different conditions — rather than us having gotten lucky on the one
run we tried.

**2. Run the "naive trigger" comparison the same number of times, on the same starting
points.**
This is the project's main claim: that being selective about *when* to fix
communication — only stepping in when communication itself looks like the problem — is
better than a blunt version that fixes things whenever the score drops, for any reason.
Right now the smart version and the naive version made the exact same call on the one test
we ran, so there's currently zero evidence this claim is true. It might be — we just
haven't tested a case where the two versions would actually disagree yet.

**3. Run the "generic fix" comparison the same number of times, on the same starting
points.**
The system's normal fix only retrains the small part of the network responsible for
communication — what an agent says and what its messages mean. The comparison version
retrains a different, similarly small part of the network that has nothing to do with
communication, as a fairness check. If the generic fix works just as well, that would mean
the recovery isn't really about "fixing communication" — it would just show that
retraining *any* similar-sized piece helps. On the one test so far, the real fix did
slightly better, but not by enough to be sure it's a real difference and not luck.

**4. Run the "no fix at all" comparison the same number of times, on the same starting
points.**
This checks what happens if nothing is done after the environment breaks, so we have a
solid "before" number to measure every fix against — confirming that number is stable and
not just noise from one lucky or unlucky run.

**5. Put all of those results together and check if the differences are real, not just
luck.**
With only one run per comparison, we can't tell a genuine pattern from random chance. This
step takes the repeated runs from items 1-4 and actually checks — using proper
statistics — whether "fixing communication beats a generic fix" or "being selective beats
the naive version" are real effects, or just how those particular single runs happened to
go. Without this step, none of the repeated running matters, no matter how many times it's
done.

**6. Check whether the cutoff numbers used throughout the system are actually reasonable,
not just convenient guesses.** *(elevated — do this right after item 5, not last)*
There are eight different cutoff numbers scattered through the system — for example, "the
score has to drop by at least 15% before we suspect a problem," or "at least 30% of the
loss has to be recovered before a fix counts as accepted." Every one of them was picked as
a round, reasonable-sounding number, never tested. On the one real run we have, the score
drop landed at 47% — just 3 points from the 50% cutoff that decides whether the system
tries a small fix first or jumps straight to the biggest one. A few points either way would
have told a completely different story. Once items 1-4 produce several repeated runs, this
step goes back through those cutoff numbers and checks whether the results would have come
out differently with slightly different cutoffs — telling us whether the current numbers
are actually reasonable or just happened to work for the handful of runs tried so far. This
can't run before items 1-4 exist (it needs their output to check against), but it matters
enough that it shouldn't be left until last either — a wrong cutoff quietly affects every
other result on this list, so it's worth confirming early.

**7. Repeat the same full comparison on a different, already-trained agent.**
Everything tested so far has used one specific trained agent. Running the same comparisons
on a second, independently trained agent checks whether the results are actually about how
the *system* works in general, or whether they just happened to be true for this one agent.

**8. Actually test the part of the system that decides how big a fix to try, instead of
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
been given a chance to choose differently. Placed last because, unlike items 1-7, it needs
genuinely new experiments (a different break strength), not just repeats of what exists.
