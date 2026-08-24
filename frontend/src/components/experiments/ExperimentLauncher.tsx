import React, { useState } from 'react';
import { Play, RotateCcw, AlertTriangle, Users, Target, Zap } from 'lucide-react';
import { RunConfig } from '../../types';

interface ExperimentLauncherProps {
  onLaunch: (config: RunConfig) => Promise<void>;
  isLaunching: boolean;
  isBusy: boolean;
}

const DEFAULT_CONFIG: RunConfig = {
  experiment_name: 'mpe_spread_experiment',
  num_agents: 2,
  num_landmarks: 3,
  seed: 1,
  num_env_steps: 100000,
  episode_length: 25,
  n_rollout_threads: 32,
  eval_interval: 5,
  disable_messages: false,
  eval_disable_messages: false,
  eval_noise_std: 0.25,
  use_eval: true,
};

export const ExperimentLauncher: React.FC<ExperimentLauncherProps> = ({
  onLaunch,
  isLaunching,
  isBusy,
}) => {
  const [config, setConfig] = useState<RunConfig>(DEFAULT_CONFIG);
  const [activePreset, setActivePreset] = useState<'2agents' | '3agents' | 'ablation' | 'custom'>('2agents');

  const applyPreset = (preset: '2agents' | '3agents' | 'ablation') => {
    setActivePreset(preset);
    if (preset === '2agents') {
      setConfig({
        ...DEFAULT_CONFIG,
        experiment_name: 'spread_2agents',
        num_agents: 2,
        num_landmarks: 3,
        num_env_steps: 100000,
      });
    } else if (preset === '3agents') {
      setConfig({
        ...DEFAULT_CONFIG,
        experiment_name: 'spread_3agents_attention',
        num_agents: 3,
        num_landmarks: 3,
        num_env_steps: 200000,
      });
    } else if (preset === 'ablation') {
      setConfig({
        ...DEFAULT_CONFIG,
        experiment_name: 'spread_no_comm_ablation',
        disable_messages: true,
        eval_disable_messages: true,
      });
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onLaunch(config);
  };

  return (
    <form className="experiment-form" onSubmit={handleSubmit}>
      {/* Presets Bar */}
      <div className="presets-bar">
        <span className="preset-title">Quick Presets:</span>
        <button
          type="button"
          className={`preset-chip ${activePreset === '2agents' ? 'active' : ''}`}
          onClick={() => applyPreset('2agents')}
        >
          <Users size={14} /> 2 Agents Baseline
        </button>
        <button
          type="button"
          className={`preset-chip ${activePreset === '3agents' ? 'active' : ''}`}
          onClick={() => applyPreset('3agents')}
        >
          <Zap size={14} /> 3 Agents (Attention Multi-Agent)
        </button>
        <button
          type="button"
          className={`preset-chip ${activePreset === 'ablation' ? 'active' : ''}`}
          onClick={() => applyPreset('ablation')}
        >
          <AlertTriangle size={14} /> No-Comm Ablation
        </button>
      </div>

      <div className="form-grid">
        {/* Core Parameters */}
        <div className="form-section">
          <h4 className="section-title">Environment & Multi-Agent Setup</h4>
          
          <div className="form-group">
            <label>Experiment Tag / Folder Name</label>
            <input
              type="text"
              value={config.experiment_name}
              onChange={(e) => {
                setConfig({ ...config, experiment_name: e.target.value });
                setActivePreset('custom');
              }}
              required
              className="form-input"
            />
          </div>

          <div className="form-row">
            <div className="form-group flex-1">
              <label><Users size={14} className="inline mr-1" /> Number of Agents</label>
              <input
                type="number"
                min={2}
                max={6}
                value={config.num_agents}
                onChange={(e) => {
                  setConfig({ ...config, num_agents: parseInt(e.target.value) || 2 });
                  setActivePreset('custom');
                }}
                required
                className="form-input"
              />
              <span className="field-hint">2 agents (standard) or 3+ for active attention</span>
            </div>

            <div className="form-group flex-1">
              <label><Target size={14} className="inline mr-1" /> Landmarks</label>
              <input
                type="number"
                min={2}
                max={8}
                value={config.num_landmarks}
                onChange={(e) => {
                  setConfig({ ...config, num_landmarks: parseInt(e.target.value) || 3 });
                  setActivePreset('custom');
                }}
                required
                className="form-input"
              />
              <span className="field-hint">Target locations to coordinate & cover</span>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group flex-1">
              <label>Random Seed</label>
              <input
                type="number"
                value={config.seed}
                onChange={(e) => setConfig({ ...config, seed: parseInt(e.target.value) || 1 })}
                required
                className="form-input"
              />
            </div>
            <div className="form-group flex-1">
              <label>Episode Max Length</label>
              <input
                type="number"
                value={config.episode_length}
                onChange={(e) => setConfig({ ...config, episode_length: parseInt(e.target.value) || 25 })}
                required
                className="form-input"
              />
            </div>
          </div>
        </div>

        {/* Training & Rollout Parameters */}
        <div className="form-section">
          <h4 className="section-title">Training Optimization & Rollout</h4>

          <div className="form-row">
            <div className="form-group flex-1">
              <label>Total Environment Steps</label>
              <input
                type="number"
                step={10000}
                value={config.num_env_steps}
                onChange={(e) => setConfig({ ...config, num_env_steps: parseInt(e.target.value) || 100000 })}
                required
                className="form-input"
              />
              <span className="field-hint">e.g. 100k for quick check, 2M for full training</span>
            </div>

            <div className="form-group flex-1">
              <label>Parallel Rollout Threads</label>
              <input
                type="number"
                value={config.n_rollout_threads}
                onChange={(e) => setConfig({ ...config, n_rollout_threads: parseInt(e.target.value) || 32 })}
                required
                className="form-input"
              />
            </div>
          </div>

          <div className="form-group">
            <label>Causal Eval Interval (Episodes)</label>
            <input
              type="number"
              value={config.eval_interval}
              onChange={(e) => setConfig({ ...config, eval_interval: parseInt(e.target.value) || 5 })}
              required
              className="form-input"
            />
          </div>

          <div className="checkbox-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={config.use_eval}
                onChange={(e) => setConfig({ ...config, use_eval: e.target.checked })}
              />
              <div>
                <strong>Enable Causal Evaluation</strong>
                <p>Log counterfactual ablated message metrics to causal_influence.csv</p>
              </div>
            </label>

            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={config.disable_messages}
                onChange={(e) => setConfig({ ...config, disable_messages: e.target.checked })}
              />
              <div>
                <strong>Ablate Training Communication</strong>
                <p>Silences message passing during training (Ablation study control)</p>
              </div>
            </label>
          </div>
        </div>
      </div>

      {/* Action Footer */}
      <div className="form-actions">
        <button
          type="button"
          className="btn-secondary"
          onClick={() => {
            setConfig(DEFAULT_CONFIG);
            setActivePreset('2agents');
          }}
          disabled={isLaunching || isBusy}
        >
          <RotateCcw size={16} /> Reset
        </button>

        <button
          type="submit"
          className="btn-primary"
          disabled={isLaunching || isBusy}
        >
          <Play size={16} />
          {isLaunching ? 'Initializing Process...' : isBusy ? 'Subprocess Busy' : 'Start Simulation & Training'}
        </button>
      </div>
    </form>
  );
};
