import React, { useState } from 'react';
import { 
  Wrench, 
  ShieldAlert, 
  Play, 
  CheckCircle2, 
  XCircle, 
  ArrowRight,
  Zap,
  Sliders
} from 'lucide-react';
import { RepairConfig, CheckpointInfo } from '../../types';
import { CheckpointSelector } from './CheckpointSelector';
import { TerminalLogViewer } from '../visualizer/TerminalLogViewer';

interface RepairDashboardProps {
  runId?: string;
  checkpoints: CheckpointInfo[];
  repairLogs: string;
  onStartRepair: (config: RepairConfig) => Promise<void>;
  isRepairing: boolean;
}

export const RepairDashboard: React.FC<RepairDashboardProps> = ({
  runId,
  checkpoints,
  repairLogs,
  onStartRepair,
  isRepairing,
}) => {
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>(
    checkpoints[0]?.name || 'checkpoint_best'
  );
  const [mirrorScope, setMirrorScope] = useState<'partner_full' | 'partner' | 'all'>('partner_full');
  const [repairTarget, setRepairTarget] = useState<'auto' | 'embedding' | 'comm' | 'full' | 'noncomm'>('auto');
  const [controller, setController] = useState<'causal' | 'reward_only'>('causal');
  const [measureEpisodes, setMeasureEpisodes] = useState<number>(6);
  const [repairIters, setRepairIters] = useState<number>(15);

  const handleLaunch = (e: React.FormEvent) => {
    e.preventDefault();
    onStartRepair({
      checkpoint_name: selectedCheckpoint,
      mirror_scope: mirrorScope,
      repair_target: repairTarget,
      controller: controller,
      measure_episodes: measureEpisodes,
      repair_iters: repairIters,
      seed: 1,
    });
  };

  // Parse logs for real-time verdict badges
  const hasAccepted = repairLogs.includes('ACCEPTED');
  const hasRejected = repairLogs.includes('REJECTED');
  const hasHeldoutConfirm = repairLogs.includes('HELD-OUT CONFIRMS');
  const hasHeldoutFailed = repairLogs.includes('HELD-OUT DOES NOT CONFIRM');

  return (
    <div className="repair-dashboard-container">
      <div className="repair-hero">
        <div className="hero-text">
          <h2>
            <Wrench className="inline mr-2 text-primary" size={24} /> Phase 2/3: Causal Breakdown & Online Repair {runId ? `— ${runId}` : ''}
          </h2>
          <p>
            Inject controlled coordinate perturbations, detect causal communication failures, execute surgical parameter fine-tuning, and verify generalization on held-out episodes.
          </p>
        </div>
      </div>

      <div className="repair-layout-grid">
        {/* Left Column: Configuration Form & Checkpoints */}
        <div className="repair-config-column">
          <form className="repair-form-card" onSubmit={handleLaunch}>
            <h4 className="card-section-title">
              <Sliders size={16} /> 1. Perturbation & Controller Setup
            </h4>

            {/* Checkpoint selector */}
            <CheckpointSelector
              checkpoints={checkpoints}
              selectedCheckpoint={selectedCheckpoint}
              onSelectCheckpoint={setSelectedCheckpoint}
            />

            {/* Perturbation Scope */}
            <div className="form-group mt-4">
              <label>
                <ShieldAlert size={14} className="inline mr-1 text-rose" /> Perturbation Modality (mirror_scope)
              </label>
              <select
                value={mirrorScope}
                onChange={(e) => setMirrorScope(e.target.value as any)}
                className="form-select"
              >
                <option value="partner_full">partner_full (180° partner perception inversion - recommended)</option>
                <option value="partner">partner (x-axis partner position negation)</option>
                <option value="all">all (full coordinate inversion - navigation breakdown)</option>
              </select>
              <span className="field-hint">
                Inverts partner relative coordinates while keeping landmark navigation mechanics intact.
              </span>
            </div>

            {/* Repair Target Slicing */}
            <div className="form-group">
              <label>
                <Wrench size={14} className="inline mr-1 text-emerald" /> Repair Parameter Target
              </label>
              <select
                value={repairTarget}
                onChange={(e) => setRepairTarget(e.target.value as any)}
                className="form-select"
              >
                <option value="auto">auto (Adaptive Controller: embedding → comm → full)</option>
                <option value="embedding">embedding only (~320 params, word meanings)</option>
                <option value="comm">comm only (message_head + token_embedding + attention_weight)</option>
                <option value="full">full (entire actor policy network)</option>
                <option value="noncomm">noncomm (Control Baseline: action_out slice)</option>
              </select>
            </div>

            {/* Trigger Controller Mode */}
            <div className="form-group">
              <label>Detection Trigger Controller</label>
              <div className="radio-pill-group">
                <button
                  type="button"
                  className={`radio-pill ${controller === 'causal' ? 'active' : ''}`}
                  onClick={() => setController('causal')}
                >
                  <Zap size={14} /> Smart Causal Trigger (Reward + Comm Drop)
                </button>
                <button
                  type="button"
                  className={`radio-pill ${controller === 'reward_only' ? 'active' : ''}`}
                  onClick={() => setController('reward_only')}
                >
                  Naive Trigger (Reward Drop Only &ge; 30%)
                </button>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group flex-1">
                <label>Measure Episodes</label>
                <input
                  type="number"
                  min={2}
                  max={20}
                  value={measureEpisodes}
                  onChange={(e) => setMeasureEpisodes(parseInt(e.target.value) || 6)}
                  className="form-input"
                />
              </div>
              <div className="form-group flex-1">
                <label>Online Repair Iterations</label>
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={repairIters}
                  onChange={(e) => setRepairIters(parseInt(e.target.value) || 15)}
                  className="form-input"
                />
              </div>
            </div>

            <button
              type="submit"
              className="btn-primary w-full mt-3"
              disabled={isRepairing || !selectedCheckpoint}
            >
              <Play size={16} />
              {isRepairing ? 'Executing Repair Loop...' : 'Execute Perturbation & Repair'}
            </button>
          </form>
        </div>

        {/* Right Column: Execution Live Status & Logs */}
        <div className="repair-results-column">
          {/* Status Verdicts Card */}
          <div className="verdict-summary-card">
            <h4 className="card-section-title">Repair Verdict State</h4>
            
            <div className="verdict-stepper">
              <div className="step-item active">
                <span className="step-num">1</span>
                <span>Baseline Fingerprint</span>
              </div>
              <ArrowRight size={14} className="text-gray" />
              <div className="step-item active">
                <span className="step-num">2</span>
                <span>Perturbation Applied</span>
              </div>
              <ArrowRight size={14} className="text-gray" />
              <div className="step-item active">
                <span className="step-num">3</span>
                <span>Detection & Slicing</span>
              </div>
              <ArrowRight size={14} className="text-gray" />
              <div className={`step-item ${hasAccepted ? 'success' : hasRejected ? 'danger' : ''}`}>
                <span className="step-num">4</span>
                <span>{hasAccepted ? 'Accepted' : hasRejected ? 'Rejected / Rollback' : 'Validation'}</span>
              </div>
            </div>

            {/* Held-Out Confirmation Banner */}
            {hasHeldoutConfirm && (
              <div className="heldout-banner success">
                <CheckCircle2 size={18} />
                <div>
                  <strong>HELD-OUT CONFIRMED</strong>
                  <p>Repaired communication generalization verified on fresh, unseen random layouts.</p>
                </div>
              </div>
            )}

            {hasHeldoutFailed && (
              <div className="heldout-banner danger">
                <XCircle size={18} />
                <div>
                  <strong>HELD-OUT DOES NOT CONFIRM</strong>
                  <p>Repair passed test batch but failed on held-out episodes (potential overfitting).</p>
                </div>
              </div>
            )}
          </div>

          {/* Real-time streaming log terminal */}
          <TerminalLogViewer
            logs={repairLogs}
            title="Phase 2/3 Online Repair Process Stream"
            autoScroll={true}
          />
        </div>
      </div>
    </div>
  );
};
