import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        # self.envs_contrastive = config['envs_contrastive']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        if self.all_args.algorithm_name == 'r_mappo_comm':
            from onpolicy.algorithms.r_mappo_comm.r_mappo_comm import R_MAPPO_COMM as TrainAlgo
            from onpolicy.algorithms.r_mappo_comm.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        elif self.all_args.algorithm_name == 't_mappo_comm':
            from onpolicy.algorithms.t_mappo_comm.t_mappo_comm import T_MAPPO_COMM as TrainAlgo
            from onpolicy.algorithms.t_mappo_comm.algorithm.tMAPPOPolicy import T_MAPPOPolicy as Policy
        elif self.all_args.algorithm_name == 'r_mappo':
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        elif self.all_args.algorithm_name == 'mappo':
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        elif self.all_args.algorithm_name == 'macppo':
            from onpolicy.algorithms.macppo.r_mappo_comm import R_MAPPO_COMM as TrainAlgo
            from onpolicy.algorithms.macppo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        elif self.all_args.algorithm_name == 'memo_ppo':
            from onpolicy.algorithms.memo_ppo.r_mappo_comm import R_MAPPO_COMM as TrainAlgo
            from onpolicy.algorithms.memo_ppo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        # print(share_observation_space)
        # policy network
        # print(self.envs.observation_space[0])
        # sys.exit()
        self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)

        if self.model_dir is not None:
            print(self.model_dir)
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.all_args.use_transformer_policy:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                    np.concatenate(self.buffer.seq_states_critic[-1]),
                                                    np.concatenate(self.buffer.masks[-1]))
        else:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                    np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                    np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self, tag=None):
        """
        Save policy's actor and critic networks.
        :param tag: (str or None) if given, weights are written to a `checkpoint_<tag>/`
                    subfolder (a permanent named snapshot) instead of overwriting the rolling
                    actor.pt / critic.pt. Used to capture a specific good episode for Phase 2/3.
        """
        if tag is not None:
            ckpt_dir = os.path.join(str(self.save_dir), "checkpoint_{}".format(tag))
            os.makedirs(ckpt_dir, exist_ok=True)
            save_dir = ckpt_dir
        else:
            save_dir = str(self.save_dir)
        torch.save(self.trainer.policy.actor.state_dict(), os.path.join(save_dir, "actor.pt"))
        torch.save(self.trainer.policy.critic.state_dict(), os.path.join(save_dir, "critic.pt"))
        # Persist the learnable attention weights too, so a restored policy reproduces the
        # exact message aggregation it was evaluated with.
        attn = getattr(self.trainer.policy, "attention_weight", None)
        if attn is not None:
            torch.save(attn.detach().cpu(), os.path.join(save_dir, "attention_weight.pt"))

    def _write_checkpoint_dir(self, path):
        """Write actor/critic/attention_weight into `path` under the exact filenames restore() expects."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.trainer.policy.actor.state_dict(), os.path.join(path, "actor.pt"))
        torch.save(self.trainer.policy.critic.state_dict(), os.path.join(path, "critic.pt"))
        attn = getattr(self.trainer.policy, "attention_weight", None)
        if attn is not None:
            torch.save(attn.detach().cpu(), os.path.join(path, "attention_weight.pt"))

    def _prune_best_checkpoints(self, keep):
        """
        Keep only the `keep` most recent checkpoint_best_<steps>/ folders, deleting older ones.

        Only ever removes directories in save_dir whose name matches checkpoint_best_<digits>
        exactly -- the naming this class writes itself. checkpoint_best/ (no suffix) and the
        per-eval checkpoint_<steps>/ history are never touched.
        """
        import re
        import shutil
        if keep is None or keep <= 0:
            return
        pattern = re.compile(r"^checkpoint_best_(\d+)$")
        found = []
        for name in os.listdir(str(self.save_dir)):
            m = pattern.match(name)
            if m and os.path.isdir(os.path.join(str(self.save_dir), name)):
                found.append((int(m.group(1)), name))
        for _, name in sorted(found, reverse=True)[keep:]:
            shutil.rmtree(os.path.join(str(self.save_dir), name), ignore_errors=True)

    def save_best(self, tag=None, keep=None):
        """
        Save the best-so-far snapshot, in three forms:

        1. `best_actor.pt` / `best_critic.pt` / `best_attention_weight.pt` -- flat files, kept
           for backwards compatibility. restore() CANNOT load these (it looks for actor.pt).
        2. `checkpoint_best/` -- overwritten every time a new best appears. A stable path that
           always holds the current best and can be passed straight to --model_dir.
        3. `checkpoint_best_<tag>/` -- a new folder per best, so the progression is inspectable.
           Pruned to the `keep` most recent (sliding window) so long runs don't accumulate
           dozens of snapshots.

        :param tag: (int or None) step count, used to name the per-best folder. Omit to write
                    only forms 1 and 2.
        :param keep: (int or None) sliding-window size for form 3. None/<=0 disables pruning.
        """
        torch.save(self.trainer.policy.actor.state_dict(), str(self.save_dir) + "/best_actor.pt")
        torch.save(self.trainer.policy.critic.state_dict(), str(self.save_dir) + "/best_critic.pt")
        attn = getattr(self.trainer.policy, "attention_weight", None)
        if attn is not None:
            torch.save(attn.detach().cpu(), str(self.save_dir) + "/best_attention_weight.pt")

        # Directly loadable forms.
        self._write_checkpoint_dir(os.path.join(str(self.save_dir), "checkpoint_best"))
        if tag is not None:
            self._write_checkpoint_dir(
                os.path.join(str(self.save_dir), "checkpoint_best_{}".format(tag)))
            self._prune_best_checkpoints(keep)

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
        # Restore learnable message-aggregation weights if the snapshot has them.
        attn_path = os.path.join(str(self.model_dir), "attention_weight.pt")
        if os.path.exists(attn_path) and getattr(self.policy, "attention_weight", None) is not None:
            with torch.no_grad():
                self.policy.attention_weight.copy_(torch.load(attn_path).to(self.policy.attention_weight.device))

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
