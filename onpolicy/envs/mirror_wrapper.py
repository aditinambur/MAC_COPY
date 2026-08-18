"""
One-sided mirror observation wrapper for Phase 2 / Phase 3.

This induces a controlled "environment change": the x-component of every spatial entry in
each agent's observation is negated (a left-right sensor / coordinate-frame flip), while the
*actions* the policy emits are still applied in the true world frame. The cooperative task
itself (cover the landmarks) is unchanged in difficulty, so any reward / communication-benefit
drop is attributable purely to the policy's learned observation->action mapping (and the
message semantics built on top of it) no longer matching the observations it now receives.

This is deliberately a *one-sided* mirror. A full mirror (flip the observation AND the action
interpretation) is a symmetry the task is invariant to, so a well-trained policy would not
degrade under it; that would defeat the purpose of a degradation test.

simple_spread per-agent observation layout (dim_p = 2):
    [ self_vel(2), self_pos(2), entity_pos(num_landmarks * 2),
      other_pos((num_agents - 1) * 2), comm((num_agents - 1) * dim_c) ]
The x-component is index 0 of every 2-D spatial sub-vector. The trailing `comm` block is the
MPE built-in communication channel (agents are silent here, so it is zero) and is non-spatial,
so it is left untouched.
"""

import numpy as np


def compute_mirror_indices(num_agents, num_landmarks, dim_p=2, scope="all"):
    """
    Return the observation indices to negate, for a simple_spread-style observation layout:
        [ self_vel(dim_p), self_pos(dim_p), entity_pos(num_landmarks*dim_p),
          other_pos((num_agents-1)*dim_p), comm((num_agents-1)*dim_c) ]

    scope controls how severe / targeted the perturbation is:
      - "all":          flip the x-component of every spatial sub-vector (self, landmarks,
                        partners). Corrupts the whole observation->action mapping; very severe,
                        navigation itself collapses. Use for a coarse "environment change".
      - "partner":      flip only the x-component of the other-agent relative positions.
                        Navigation to landmarks is preserved; the agent's belief about WHERE
                        its partner is gets corrupted -> coordination (which communication
                        supports) degrades. Best fit for a communication-repair story.
      - "partner_full": negate BOTH components of the other-agent relative positions (a 180°
                        flip of partner perception). Stronger than "partner", still leaves
                        self/landmark navigation intact.
    """
    self_block = 2 * dim_p                                  # self_vel + self_pos
    other_start = self_block + num_landmarks * dim_p        # start of other_pos block
    other_end = other_start + (num_agents - 1) * dim_p      # end of other_pos block (comm excluded)

    if scope == "all":
        return list(range(0, other_end, dim_p))            # x of every spatial sub-vector
    if scope == "partner":
        return list(range(other_start, other_end, dim_p))  # x of partner positions only
    if scope == "partner_full":
        return list(range(other_start, other_end))         # both comps of partner positions
    raise ValueError("Unknown mirror scope: {}".format(scope))


def compute_mirror_x_indices(num_agents, num_landmarks, dim_p=2):
    """Backwards-compatible alias for the full x-axis mirror (scope='all')."""
    return compute_mirror_indices(num_agents, num_landmarks, dim_p, scope="all")


class MirrorObsVecEnv:
    """
    Thin wrapper around a vectorized env that negates selected observation indices (the x-axis
    spatial components) on every reset() and step(). All other attributes/methods (action_space,
    observation_space, seed, close, num_envs, ...) are transparently forwarded to the inner env,
    so this can be dropped in wherever a vec env is expected (e.g. runner.eval_envs).

    :param venv: the wrapped vectorized environment.
    :param mirror_indices: (sequence[int]) observation indices to negate (last-axis indices).
    """

    def __init__(self, venv, mirror_indices):
        self.venv = venv
        self.mirror_indices = np.asarray(list(mirror_indices), dtype=np.int64)

    def _mirror(self, obs):
        obs = np.array(obs, copy=True)
        # obs shape: [n_envs, n_agents, obs_dim]; flip x components on the last axis.
        obs[..., self.mirror_indices] = -obs[..., self.mirror_indices]
        return obs

    def reset(self, *args, **kwargs):
        return self._mirror(self.venv.reset(*args, **kwargs))

    def step(self, *args, **kwargs):
        obs, rews, dones, infos = self.venv.step(*args, **kwargs)
        return self._mirror(obs), rews, dones, infos

    def __getattr__(self, name):
        # Only reached for attributes not found on the wrapper itself.
        return getattr(self.venv, name)
