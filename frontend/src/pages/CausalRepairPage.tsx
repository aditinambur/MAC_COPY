import React from 'react';
import { Wrench, ArrowLeft } from 'lucide-react';
import { Run, CheckpointInfo, RepairConfig } from '../types';
import { RepairDashboard } from '../components/repair/RepairDashboard';
import { TrajectoryCanvas } from '../components/visualizer/TrajectoryCanvas';

interface CausalRepairPageProps {
  selectedRun: Run | null;
  checkpoints: CheckpointInfo[];
  repairLogs: string;
  onStartRepair: (config: RepairConfig) => Promise<void>;
  isRepairing: boolean;
  onBack: () => void;
}

export const CausalRepairPage: React.FC<CausalRepairPageProps> = ({
  selectedRun,
  checkpoints,
  repairLogs,
  onStartRepair,
  isRepairing,
  onBack,
}) => {
  if (!selectedRun) {
    return (
      <div className="page-container">
        <div className="empty-state">
          <Wrench size={40} className="text-gray" />
          <h3>No Experiment Selected</h3>
          <p>Please select an experiment from the Dashboard or Experiments list to launch Causal Repair.</p>
          <button className="btn-primary-sm mt-3" onClick={onBack}>
            <ArrowLeft size={14} /> Go to Experiments
          </button>
        </div>
      </div>
    );
  }

  const numAgents = selectedRun.config?.num_agents || 2;
  const numLandmarks = selectedRun.config?.num_landmarks || 3;

  return (
    <div className="page-container">
      {/* Top Bar */}
      <div className="detail-header-bar">
        <button className="back-btn" onClick={onBack}>
          <ArrowLeft size={16} /> Back to Run
        </button>
        <span className="current-run-label">
          Target Experiment: <strong>{selectedRun.run_id}</strong>
        </span>
      </div>

      {/* Perturbation Trajectory Demo Canvas */}
      <div className="repair-simulation-banner">
        <TrajectoryCanvas
          numAgents={numAgents}
          numLandmarks={numLandmarks}
          isPerturbed={true}
          perturbationScope="partner_full"
        />
      </div>

      {/* Repair Execution Dashboard */}
      <RepairDashboard
        runId={selectedRun.run_id}
        checkpoints={checkpoints}
        repairLogs={repairLogs}
        onStartRepair={onStartRepair}
        isRepairing={isRepairing}
      />
    </div>
  );
};
