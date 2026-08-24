import time
import os
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
try:
    # pyrefly: ignore [missing-import]
    import wandb
except Exception:
    wandb = None
# pyrefly: ignore [missing-import]
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
        self.gif_dir = self.all_args.gif_dir if self.all_args.gif_dir is not None else '.'
        self.message_log_dir = os.path.join(str(self.run_dir), "messages")
        os.makedirs(self.message_log_dir, exist_ok=True)
        
        # Phase 3: Initialize message buffers
        self.latest_messages = None
        self.latest_aggregated_messages = None
        self.latest_policy_messages = None

        # Best-checkpoint tracking (for capturing a strong policy for Phase 2/3).
        self.best_eval_score = -np.inf
        self.latest_eval_normal_reward = None
        self.latest_eval_comm_effect = None
        self.latest_eval_value_sensitivity = None

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, reconstructions = self.collect(step)
                self._log_messages(episode, step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env, reconstructions)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, reconstructions

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

                # Capture a permanent named snapshot at every eval, plus a rolling "best"
                # snapshot, so a strong communication-reliant policy can be recovered later
                # for Phase 2 (degradation detection) / Phase 3 (online repair).
                self.save(tag=total_num_steps)
                if self.latest_eval_normal_reward is not None:
                    # Selection score rewards all three signals Phase 2/3 depend on: return,
                    # communication benefit (comm_effect), and critic value-sensitivity to
                    # messages. The 100x weight puts value-sensitivity (~O(1)) on a comparable
                    # scale to reward/comm_effect (~O(100s)), so a strong-reward-but-weak-comm
                    # snapshot does not win over a balanced one.
                    score = self.latest_eval_normal_reward
                    if self.latest_eval_comm_effect is not None:
                        score = score + max(0.0, self.latest_eval_comm_effect)
                    if self.latest_eval_value_sensitivity is not None:
                        score = score + 100.0 * self.latest_eval_value_sensitivity
                    if score > self.best_eval_score:
                        self.best_eval_score = score
                        # Writes checkpoint_best/ (stable, always the current best) and a
                        # sliding window of checkpoint_best_<steps>/ folders, both loadable
                        # directly via --model_dir. See Runner.save_best.
                        self.save_best(tag=total_num_steps,
                                       keep=getattr(self.all_args, "best_keep", 3))
                        print("[CKPT] New best eval at step {} (reward={:.1f}, comm_effect={}, value_sens={}) "
                              "-> saved best_*.pt (same weights also in checkpoint_{}/)".format(
                                  total_num_steps,
                                  self.latest_eval_normal_reward,
                                  "n/a" if self.latest_eval_comm_effect is None else "{:.1f}".format(self.latest_eval_comm_effect),
                                  "n/a" if self.latest_eval_value_sensitivity is None else "{:.3f}".format(self.latest_eval_value_sensitivity),
                                  total_num_steps))

    def _log_messages(self, episode, step):
        """Persist communication tensors for debugging under scripts/results/.../messages."""
        # Debug-only and very I/O heavy (2 files per env-step); off unless explicitly requested.
        if not getattr(self.all_args, "save_messages", False):
            return

        messages = getattr(self, "latest_messages", None)
        aggregated_messages = getattr(self, "latest_aggregated_messages", None)

        if messages is None:
            return

        base = "episode_{:06d}_step_{:04d}".format(episode, step)
        np.save(os.path.join(self.message_log_dir, base + "_messages.npy"), messages)
        if aggregated_messages is not None:
            np.save(os.path.join(self.message_log_dir, base + "_aggregated_messages.npy"), aggregated_messages)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # Phase 3: Initialize zero messages for the first step
        # so agents can start using communication from step 0
        message_dim = self.all_args.hidden_size  # Message embeddings have same dim as hidden layer
        self.latest_policy_messages = np.zeros(
            (self.n_rollout_threads * self.num_agents, self.num_agents, message_dim),
            dtype=np.float32,
        )
        self.latest_aggregated_messages = np.zeros((self.n_rollout_threads, self.num_agents, message_dim), 
                                               dtype=np.float32)

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        
        # Phase 3: Reshape aggregated messages for actor input if available
        messages_input = None
        if hasattr(self, 'latest_policy_messages') and self.latest_policy_messages is not None:
            messages_input = self.latest_policy_messages
        
        
        policy_out = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]),
                            messages=messages_input)

        messages = None
        aggregated_messages = None
        reconstructions = None

        if len(policy_out) == 6:
            value, action, action_log_prob, rnn_states, rnn_states_critic, aux_output = policy_out
            aux_np = _t2n(aux_output)

            # Phase 2: treat 6th output as agent message by default.
            # Keep compatibility for adaptive-sampling setups where the 6th output is reconstruction.
            if self.all_args.scenario_name == "simple_adaptive_sampling" and aux_np.shape[-1] == self.all_args.env_size ** 2:
                reconstructions = np.array(np.split(aux_np, self.n_rollout_threads))
            else:
                # [n_envs * n_agents, message_dim] -> [n_envs, n_agents, message_dim]
                messages = np.array(np.split(aux_np, self.n_rollout_threads))
                # PHASE 3: Compute aggregated messages per agent
                # messages shape: [n_envs, n_agents, message_dim]
                # aggregated_messages[e][i] = mean of messages from agents j!=i in env e
                n_envs, n_agents, msg_dim = messages.shape
                aggregated_messages = np.zeros((n_envs, n_agents, msg_dim), dtype=np.float32)
                for agent_i in range(n_agents):
                    # Sum messages from all agents except agent_i
                    mask = np.ones(n_agents, dtype=bool)
                    mask[agent_i] = False
                    aggregated_messages[:, agent_i, :] = messages[:, mask, :].mean(axis=1)
        elif len(policy_out) == 5:
            value, action, action_log_prob, rnn_states, rnn_states_critic = policy_out
        else:
            
            raise ValueError("Unexpected get_actions() output length: {}".format(len(policy_out)))

        # CAUSAL EXPERIMENT: Disable messages if flag is set
        if self.all_args.disable_messages and aggregated_messages is not None:
            aggregated_messages = np.zeros_like(aggregated_messages)
        
        # Optional debug visibility for message passing.
        self.latest_messages = messages
        self.latest_aggregated_messages = aggregated_messages
        if messages is not None:
            if self.all_args.disable_messages:
                self.latest_policy_messages = np.zeros(
                    (self.n_rollout_threads * self.num_agents, self.num_agents, messages.shape[-1]),
                    dtype=np.float32,
                )
            else:
                self.latest_policy_messages = self._build_receiver_message_tensor(messages)
        else:
            self.latest_policy_messages = None
        self.action_input_aggregated_messages = aggregated_messages  # Store aggregated for replay consistency

        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, reconstructions

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, reconstructions = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks,
                   reconstructions=reconstructions, aggregated_messages=getattr(self, 'action_input_aggregated_messages', None))

    def _seed_eval_envs(self, seed):
        """
        Seed the eval environments so reset() produces a deterministic initial layout.
        Used for Common Random Numbers (CRN): seeding identically before each intervention makes
        all conditions start from the SAME layout. MPE's env.seed() seeds the global NumPy RNG and
        reset() draws the layout from it, so seeding the global RNG is sufficient for in-process
        (DummyVecEnv) evaluation. (For SubprocVecEnv the seed would not cross the process boundary.)
        """
        np.random.seed(seed)
        seed_fn = getattr(self.eval_envs, "seed", None)
        if callable(seed_fn):
            try:
                seed_fn(seed)
            except Exception:
                pass

    @staticmethod
    def _build_receiver_message_tensor(messages):
        """
        Expand per-env sender messages so each receiver gets the full sender set.

        :param messages: np.ndarray shaped [n_envs, n_agents, message_dim]
        :return: np.ndarray shaped [n_envs * n_agents, n_agents, message_dim]
        """
        n_envs, n_agents, msg_dim = messages.shape
        sender_messages = np.repeat(messages[:, None, :, :], n_agents, axis=1)
        return sender_messages.reshape(n_envs * n_agents, n_agents, msg_dim).astype(np.float32, copy=False)

    @torch.no_grad()
    def _eval_with_intervention(self, intervention_type='normal', crn_seed=None):
        """
        Run a single evaluation episode with a specified message intervention, through the
        communication-aware actor AND critic pipeline.

        :param intervention_type: 'normal' (no intervention), 'no_messages' (zero the incoming
                                  aggregated message), or 'noisy' (add Gaussian noise to it).
        :param crn_seed: (int or None) if given, the eval envs are seeded with it before reset so
                         every condition starts from the SAME initial layout (Common Random Numbers).
        :return: (np.ndarray) per-step rewards, shape [episode_length, n_threads, n_agents, 1].
        """
        # Common Random Numbers: seed before reset so the initial layout is identical across
        # interventions, making reward differences attributable to the intervention, not the layout.
        if crn_seed is not None:
            self._seed_eval_envs(crn_seed)

        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # Initialize messages (zero for first step, like warmup)
        message_dim = self.all_args.hidden_size
        eval_policy_messages = np.zeros(
            (self.n_eval_rollout_threads * self.num_agents, self.num_agents, message_dim),
            dtype=np.float32,
        )

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            
            # CAUSAL INTERVENTION: applied to the incoming aggregated message BEFORE the policy.
            messages_for_policy = eval_policy_messages.copy()
            if intervention_type == 'no_messages':
                messages_for_policy = np.zeros_like(messages_for_policy)
            elif intervention_type == 'noisy':
                if self.all_args.eval_noise_std > 0:
                    noise = np.random.normal(0, self.all_args.eval_noise_std, size=messages_for_policy.shape)
                    messages_for_policy = messages_for_policy + noise
            elif intervention_type != 'normal':
                raise ValueError("Unknown intervention_type: {}".format(intervention_type))

            # Build the centralized critic obs exactly as during training (centralized-V):
            # concat all agents' obs, replicated per agent -> [n_envs*n_agents, n_agents*obs_dim].
            obs = np.concatenate(eval_obs)
            if self.use_centralized_V:
                cent_obs = eval_obs.reshape(self.n_eval_rollout_threads, -1)
                cent_obs = np.expand_dims(cent_obs, 1).repeat(self.num_agents, axis=1)
                cent_obs = np.concatenate(cent_obs)
            else:
                cent_obs = obs

            # Communication-aware action path. No silent fallback: real errors surface so a
            # broken intervention can never masquerade as a message-free rollout.
            policy_out = self.trainer.policy.get_actions(
                cent_obs,
                obs,
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
                messages=messages_for_policy
            )

            if len(policy_out) == 6:
                _, eval_action, _, eval_rnn_states_new, _, eval_messages_out = policy_out
                eval_messages = _t2n(eval_messages_out)
            elif len(policy_out) == 5:
                _, eval_action, _, eval_rnn_states_new, _ = policy_out
                eval_messages = None
            else:
                raise ValueError("Unexpected get_actions() output length: {}".format(len(policy_out)))

            eval_reconstructions = None
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states_new), self.n_eval_rollout_threads))
            eval_action = _t2n(eval_action)

            eval_actions = np.array(np.split(eval_action, self.n_eval_rollout_threads))
            
            # Convert actions to environment format
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Step environment
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions_env, 
                eval_reconstructions if eval_reconstructions is not None else None
            )
            eval_episode_rewards.append(eval_rewards)

            # Reset states for done environments
            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
            # Compute aggregated messages for next step
            if eval_messages is not None:
                # eval_messages shape: [n_envs*n_agents, message_dim]
                # Reshape to [n_envs, n_agents, message_dim], aggregate, reshape back
                n_envs = self.n_eval_rollout_threads
                n_agents = self.num_agents
                msg_dim = eval_messages.shape[-1]
                
                messages_reshaped = eval_messages.reshape(n_envs, n_agents, msg_dim)
                eval_policy_messages = self._build_receiver_message_tensor(messages_reshaped)
            else:
                # No messages available, reset to zeros
                eval_policy_messages = np.zeros(
                    (self.n_eval_rollout_threads * self.num_agents, self.num_agents, message_dim),
                    dtype=np.float32,
                )

        return np.array(eval_episode_rewards)

    @staticmethod
    def _action_kl(dist_p, dist_q):
        """
        KL divergence between two action distributions (or lists of distributions, for
        multi_discrete / mixed action spaces), summed over action dimensions.
        :return: (torch.Tensor) [batch_size] KL divergence per sample.
        """
        if isinstance(dist_p, list):
            kl = None
            for p, q in zip(dist_p, dist_q):
                k = torch.distributions.kl_divergence(p, q)
                if k.dim() > 1:
                    k = k.sum(-1)
                kl = k if kl is None else kl + k
            return kl
        else:
            kl = torch.distributions.kl_divergence(dist_p, dist_q)
            if kl.dim() > 1:
                kl = kl.sum(-1)
            return kl

    @torch.no_grad()
    def _eval_causal_influence(self, crn_seed=None):
        """
        Causal Influence of Communication (CIC): for each agent, measure the effect of an
        intervention that ablates (zeroes) its incoming aggregated message on (a) its action
        distribution (KL divergence vs. the non-intervened distribution) and (b) the critic's
        value estimate (|value_real - value_ablated|). The trajectory itself follows the
        normal (non-intervened) policy; the intervention is only used to measure sensitivity
        at each visited state.

        :return: (dict) per-agent and mean KL / value-sensitivity arrays.
        """
        n_envs = self.n_eval_rollout_threads
        n_agents = self.num_agents
        message_dim = self.all_args.hidden_size

        if crn_seed is not None:
            self._seed_eval_envs(crn_seed)

        eval_obs = self.eval_envs.reset()
        eval_rnn_states = np.zeros((n_envs, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((n_envs, n_agents, 1), dtype=np.float32)
        eval_policy_messages = np.zeros((n_envs * n_agents, n_agents, message_dim), dtype=np.float32)

        kl_sums = np.zeros(n_agents, dtype=np.float64)
        value_sensitivity_sums = np.zeros(n_agents, dtype=np.float64)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()

            obs = np.concatenate(eval_obs)
            if self.use_centralized_V:
                cent_obs = eval_obs.reshape(n_envs, -1)
                cent_obs = np.expand_dims(cent_obs, 1).repeat(n_agents, axis=1)
                cent_obs = np.concatenate(cent_obs)
            else:
                cent_obs = obs
            rnn_states = np.concatenate(eval_rnn_states)
            masks = np.concatenate(eval_masks)
            zero_messages = np.zeros_like(eval_policy_messages)

            # CAUSAL INTERVENTION: ablate each agent's incoming message and measure the effect
            # on its action distribution and value estimate, without altering the trajectory.
            dist_real = self.trainer.policy.get_action_distribution(obs, rnn_states, masks, messages=eval_policy_messages)
            dist_zero = self.trainer.policy.get_action_distribution(obs, rnn_states, masks, messages=zero_messages)
            kl = self._action_kl(dist_real, dist_zero).detach().cpu().numpy().reshape(n_envs, n_agents)
            kl_sums += kl.sum(axis=0)

            values_real = self.trainer.policy.get_values(cent_obs, rnn_states, masks, messages=eval_policy_messages)
            values_zero = self.trainer.policy.get_values(cent_obs, rnn_states, masks, messages=zero_messages)
            value_delta = (values_real - values_zero).abs().detach().cpu().numpy().reshape(n_envs, n_agents)
            value_sensitivity_sums += value_delta.sum(axis=0)

            # Advance the trajectory using the normal (non-intervened) policy.
            policy_out = self.trainer.policy.get_actions(
                cent_obs, obs, rnn_states, rnn_states, masks,
                deterministic=True, messages=eval_policy_messages)

            if len(policy_out) == 6:
                _, eval_action, _, eval_rnn_states_new, _, eval_messages_out = policy_out
                eval_messages = _t2n(eval_messages_out)
            else:
                _, eval_action, _, eval_rnn_states_new, _ = policy_out
                eval_messages = None

            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states_new), n_envs))
            eval_action = _t2n(eval_action)
            eval_actions = np.array(np.split(eval_action, n_envs))

            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i] + 1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            eval_obs, _, eval_dones, _ = self.eval_envs.step(eval_actions_env)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((n_envs, n_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

            if eval_messages is not None:
                messages_reshaped = eval_messages.reshape(n_envs, n_agents, message_dim)
                eval_policy_messages = self._build_receiver_message_tensor(messages_reshaped)
            else:
                eval_policy_messages = np.zeros((n_envs * n_agents, n_agents, message_dim), dtype=np.float32)

        kl_mean = kl_sums / self.episode_length
        value_sensitivity_mean = value_sensitivity_sums / self.episode_length

        results = {}
        for agent_i in range(n_agents):
            results['causal_influence_kl_agent%d' % agent_i] = np.array([kl_mean[agent_i]])
            results['causal_influence_value_sensitivity_agent%d' % agent_i] = np.array([value_sensitivity_mean[agent_i]])
        results['causal_influence_kl_mean'] = np.array([kl_mean.mean()])
        results['causal_influence_value_sensitivity_mean'] = np.array([value_sensitivity_mean.mean()])
        return results

    def _save_causal_influence_csv(self, causal_influence_infos, total_num_steps):
        """
        Append the causal-influence metrics for this eval to a human-readable CSV under the
        run directory, so results persist regardless of wandb/tensorboard.
        File: <run_dir>/causal_influence.csv
        """
        import csv

        csv_path = os.path.join(str(self.run_dir), "causal_influence.csv")
        # Stable column order: total_num_steps first, then metric keys sorted.
        metric_keys = sorted(causal_influence_infos.keys())
        row = {"total_num_steps": total_num_steps}
        for k in metric_keys:
            val = causal_influence_infos[k]
            row[k] = float(np.asarray(val).reshape(-1)[0])

        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["total_num_steps"] + metric_keys)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print("[EVAL] Causal-influence metrics appended to {}".format(csv_path))

    @torch.no_grad()
    def eval(self, total_num_steps):
        """
        Run evaluation with optional message interventions. The normal / no-message / noisy
        conditions are run as Common-Random-Number-paired episodes (same initial layouts), and
        averaged over several episodes to reduce variance, so the reward gaps between conditions
        are causally attributable to the intervention rather than to the random layout.
        """
        eval_env_infos = {}

        n_crn_episodes = max(1, int(getattr(self.all_args, "eval_crn_episodes", 1)))
        base_seed = self.all_args.seed * 100003 + int(total_num_steps)

        run_no_msg = self.all_args.eval_disable_messages
        run_noisy = self.all_args.eval_noise_std > 0

        normal_returns, no_msg_returns, noisy_returns = [], [], []

        print("\n[EVAL] Running {} CRN-paired episode(s) per condition...".format(n_crn_episodes))
        for k in range(n_crn_episodes):
            crn_seed = base_seed + k  # SAME seed reused across all conditions in this episode

            normal_returns.append(np.sum(
                self._eval_with_intervention(intervention_type='normal', crn_seed=crn_seed), axis=0))

            if run_no_msg:
                no_msg_returns.append(np.sum(
                    self._eval_with_intervention(intervention_type='no_messages', crn_seed=crn_seed), axis=0))

            if run_noisy:
                noisy_returns.append(np.sum(
                    self._eval_with_intervention(intervention_type='noisy', crn_seed=crn_seed), axis=0))

        normal_returns = np.array(normal_returns)
        eval_env_infos['eval_normal_rewards'] = np.mean(normal_returns, axis=0)
        # Scalar summaries used by run() for best-checkpoint selection (Phase 2/3 needs a
        # policy that both scores well AND visibly relies on communication).
        self.latest_eval_normal_reward = float(np.mean(normal_returns))
        self.latest_eval_comm_effect = None

        if run_no_msg:
            no_msg_returns = np.array(no_msg_returns)
            eval_env_infos['eval_no_message_rewards'] = np.mean(no_msg_returns, axis=0)
            # CRN-controlled causal effect of communication on return (paired per episode).
            eval_env_infos['eval_comm_effect_vs_no_message'] = np.mean(normal_returns - no_msg_returns, axis=0)
            self.latest_eval_comm_effect = float(np.mean(normal_returns - no_msg_returns))

        if run_noisy:
            noisy_returns = np.array(noisy_returns)
            eval_env_infos['eval_noisy_message_rewards'] = np.mean(noisy_returns, axis=0)
            eval_env_infos['eval_comm_effect_vs_noisy'] = np.mean(normal_returns - noisy_returns, axis=0)

        # Causal Influence of Communication (CIC): per-agent KL divergence and value
        # sensitivity when each agent's incoming message is ablated.
        if self.all_args.eval_causal_influence:
            print("\n[EVAL] Measuring causal influence of communication...")
            causal_influence_infos = self._eval_causal_influence()
            eval_env_infos.update(causal_influence_infos)

            csv_metrics = dict(causal_influence_infos)
            if 'eval_normal_rewards' in eval_env_infos:
                csv_metrics['eval_reward'] = np.array([float(np.mean(eval_env_infos['eval_normal_rewards']))])
            if 'eval_comm_effect_vs_no_message' in eval_env_infos:
                csv_metrics['comm_effect'] = np.array([float(np.mean(eval_env_infos['eval_comm_effect_vs_no_message']))])
            elif 'eval_comm_effect_vs_noisy' in eval_env_infos:
                csv_metrics['comm_effect'] = np.array([float(np.mean(eval_env_infos['eval_comm_effect_vs_noisy']))])

            self._save_causal_influence_csv(csv_metrics, total_num_steps)
            # Expose value-sensitivity for best-checkpoint selection (Phase 2 keys on it).
            self.latest_eval_value_sensitivity = float(
                np.asarray(causal_influence_infos['causal_influence_value_sensitivity_mean']).reshape(-1)[0])

        # Print and log results
        print("\n[EVAL RESULTS]")
        for key, val in eval_env_infos.items():
            print("{}: {}".format(key, val))
        
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        reconstructed_frames = [[] for i in range(self.all_args.num_agents)]
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                imgs = envs.render('rgb_array')
                image = imgs[0][0]
                all_frames.append(image)
                if self.all_args.scenario_name == "simple_adaptive_sampling":
                    for i in range(self.all_args.num_agents):
                        r_img = imgs[0][1][i]
                        reconstructed_frames[i].append(r_img)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                render_out = self.trainer.policy.act(np.concatenate(obs),
                                                     np.concatenate(rnn_states),
                                                     np.concatenate(masks),
                                                     deterministic=True)

                if len(render_out) == 3:
                    action, rnn_states, reconstruction = render_out
                    reconstructions = np.array(np.split(_t2n(reconstruction), self.n_rollout_threads))
                elif len(render_out) == 2:
                    action, rnn_states = render_out
                    reconstructions = None
                else:
                    raise ValueError("Unexpected act() output length: {}".format(len(render_out)))

                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env, reconstructions)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    imgs = envs.render('rgb_array')
                    image = imgs[0][0]
                    all_frames.append(image)
                    if self.all_args.scenario_name == "simple_adaptive_sampling":
                        for i in range(self.all_args.num_agents):
                            r_img = imgs[0][1][i]
                            reconstructed_frames[i].append(r_img)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
            for i in range(self.all_args.num_agents):
                imageio.mimsave(str(self.gif_dir) + '/render_reconstruction_agent' + str(i) + '.gif', reconstructed_frames[i], duration=self.all_args.ifi)

