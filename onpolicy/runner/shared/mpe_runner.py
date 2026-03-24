import time
import os
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
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
        self.latest_shared_messages = None
        self.action_input_shared_messages = None

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

    def _log_messages(self, episode, step):
        """Persist communication tensors for debugging under scripts/results/.../messages."""
        messages = getattr(self, "latest_messages", None)
        shared_messages = getattr(self, "latest_shared_messages", None)

        if messages is None:
            return

        base = "episode_{:06d}_step_{:04d}".format(episode, step)
        np.save(os.path.join(self.message_log_dir, base + "_messages.npy"), messages)
        if shared_messages is not None:
            np.save(os.path.join(self.message_log_dir, base + "_shared_messages.npy"), shared_messages)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # Phase 3: Initialize zero messages for the first step
        # so agents can start using communication from step 0
        message_dim = self.all_args.hidden_size  # Message embeddings have same dim as hidden layer
        self.latest_shared_messages = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents, message_dim), 
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
        
        # Phase 3: Reshape messages for actor input if available
        messages_input = None
        action_input_shared_messages = None
        if hasattr(self, 'latest_shared_messages') and self.latest_shared_messages is not None:
            # Keep the exact shared tensor used to pick this step's actions for PPO replay consistency.
            action_input_shared_messages = self.latest_shared_messages.copy()
            # latest_shared_messages shape: [n_envs, n_agents, n_agents, message_dim]
            # Reshape for batched policy call: [n_envs*n_agents, n_agents, message_dim]
            n_envs = self.latest_shared_messages.shape[0]
            n_agents = self.latest_shared_messages.shape[1]
            message_dim = self.latest_shared_messages.shape[3]
            messages_input = self.latest_shared_messages.reshape(n_envs * n_agents, n_agents, message_dim)
        
        policy_out = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]),
                            messages=messages_input)

        messages = None
        shared_messages = None
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
                # [n_envs, n_agents, message_dim] -> [n_envs, n_agents, n_agents, message_dim]
                # shared_messages[e][i] contains all agent messages within env e.
                shared_messages = np.repeat(messages[:, None, :, :], self.num_agents, axis=1)
        elif len(policy_out) == 5:
            value, action, action_log_prob, rnn_states, rnn_states_critic = policy_out
        else:
            
            raise ValueError("Unexpected get_actions() output length: {}".format(len(policy_out)))

        # Optional debug visibility for message passing (not used by PPO update yet).
        self.latest_messages = messages
        self.latest_shared_messages = shared_messages
        self.action_input_shared_messages = action_input_shared_messages

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
                   reconstructions=reconstructions, shared_messages=self.action_input_shared_messages)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_out = self.trainer.policy.act(np.concatenate(eval_obs),
                                               np.concatenate(eval_rnn_states),
                                               np.concatenate(eval_masks),
                                               deterministic=True)

            if len(eval_out) == 3:
                eval_action, eval_rnn_states, reconstruction = eval_out
                eval_reconstructions = np.array(np.split(_t2n(reconstruction), self.n_eval_rollout_threads))
            elif len(eval_out) == 2:
                eval_action, eval_rnn_states = eval_out
                eval_reconstructions = None
            else:
                raise ValueError("Unexpected act() output length: {}".format(len(eval_out)))

            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
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

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env, eval_reconstructions)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        print("eval average episode rewards of agent: " + str(eval_env_infos['eval_average_episode_rewards']))
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

