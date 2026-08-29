export interface Run {
  run_id: string;
  experiment_name: string;
  run_name: string;
  path: string;
  status: 'running' | 'completed' | 'failed' | 'stopped' | 'unknown';
  config: RunConfig;
  has_metrics: boolean;
  has_gif: boolean;
  has_repair: boolean;
  checkpoints_count: number;
  archived: boolean;
}

export interface RunConfig {
  experiment_name: string;
  seed: number;
  num_agents: number;
  num_landmarks: number;
  num_env_steps: number;
  episode_length: number;
  n_rollout_threads: number;
  eval_interval: number;
  disable_messages: boolean;
  eval_disable_messages: boolean;
  eval_noise_std: number;
  use_eval: boolean;
}

export interface RepairConfig {
  checkpoint_name: string;
  mirror_scope: 'partner_full' | 'partner' | 'all';
  repair_target: 'auto' | 'embedding' | 'comm' | 'full' | 'noncomm';
  controller: 'causal' | 'reward_only';
  measure_episodes: number;
  repair_iters: number;
  seed: number;
}

export interface SystemInfo {
  cuda_available: boolean;
  device_count: number;
  device_name: string;
  torch_version: string;
  active_run_id: string | null;
  process_type: 'training' | 'repair' | null;
  status: 'busy' | 'idle';
}

export interface MetricRow {
  step?: number;
  episode?: number;
  reward?: number;
  eval_reward?: number;
  causal_influence_kl_mean?: number;
  causal_influence_value_sensitivity_mean?: number;
  comm_effect?: number;
  [key: string]: any;
}

export interface CheckpointInfo {
  name: string;
  path: string;
  step: number;
  is_best: boolean;
}

export interface TrajectoryStep {
  step: number;
  agents: {
    id: number;
    x: number;
    y: number;
    vx: number;
    vy: number;
    emitted_token: number; // 0 - 4
    perceived_partner_x?: number; // perturbed position
    perceived_partner_y?: number;
  }[];
  landmarks: {
    id: number;
    x: number;
    y: number;
  }[];
}
