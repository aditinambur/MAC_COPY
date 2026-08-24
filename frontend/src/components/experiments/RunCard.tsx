import React from 'react';
import { Eye, Wrench, Archive, Trash2, Layers, Cpu } from 'lucide-react';
import { Run } from '../../types';
import { Badge } from '../common/Badge';

interface RunCardProps {
  run: Run;
  isSelected: boolean;
  onSelect: (runId: string) => void;
  onInspectRepair: (runId: string) => void;
  onToggleArchive: (runId: string) => void;
  onDelete: (runId: string) => void;
}

export const RunCard: React.FC<RunCardProps> = ({
  run,
  isSelected,
  onSelect,
  onInspectRepair,
  onToggleArchive,
  onDelete,
}) => {
  const numAgents = run.config?.num_agents || 2;
  const numLandmarks = run.config?.num_landmarks || 3;
  const steps = run.config?.num_env_steps?.toLocaleString() || '100k';

  return (
    <div className={`run-card ${isSelected ? 'selected' : ''}`}>
      <div className="run-card-header">
        <div className="run-identity">
          <h4 className="run-name">{run.run_id}</h4>
          <span className="run-path">{run.path}</span>
        </div>
        <Badge variant={run.status}>{run.status.toUpperCase()}</Badge>
      </div>

      <div className="run-card-body">
        <div className="run-meta-badges">
          <span className="meta-badge">
            <Layers size={13} /> {numAgents} Agents / {numLandmarks} Targets
          </span>
          <span className="meta-badge">
            <Cpu size={13} /> {steps} steps
          </span>
          {run.checkpoints_count > 0 && (
            <span className="meta-badge highlight">
              {run.checkpoints_count} Checkpoints
            </span>
          )}
          {run.has_repair && (
            <span className="meta-badge repair">
              Repaired
            </span>
          )}
        </div>
      </div>

      <div className="run-card-footer">
        <button
          className="btn-outline-sm"
          onClick={() => onSelect(run.run_id)}
          title="View metrics, logs, and trajectories"
        >
          <Eye size={14} /> Details
        </button>

        <button
          className="btn-outline-sm"
          onClick={() => onInspectRepair(run.run_id)}
          title="Open Causal Repair & Breakdown Studio"
        >
          <Wrench size={14} /> Causal Repair
        </button>

        <button
          className="icon-btn-sm"
          onClick={() => onToggleArchive(run.run_id)}
          title={run.archived ? 'Unarchive' : 'Archive run'}
        >
          <Archive size={14} />
        </button>

        <button
          className="icon-btn-sm danger"
          onClick={() => onDelete(run.run_id)}
          title="Delete run directory"
        >
          <Trash2 size={14} />
        </button>
      </div>
    </div>
  );
};
