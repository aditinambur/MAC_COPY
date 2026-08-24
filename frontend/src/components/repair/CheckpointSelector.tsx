import React from 'react';
import { Layers, AlertTriangle, CheckCircle, Award } from 'lucide-react';
import { CheckpointInfo } from '../../types';

interface CheckpointSelectorProps {
  checkpoints: CheckpointInfo[];
  selectedCheckpoint: string;
  onSelectCheckpoint: (name: string) => void;
}

export const CheckpointSelector: React.FC<CheckpointSelectorProps> = ({
  checkpoints,
  selectedCheckpoint,
  onSelectCheckpoint,
}) => {
  if (!checkpoints || checkpoints.length === 0) {
    return (
      <div className="empty-checkpoints-box">
        <p>No model checkpoints found in models/ directory. Run training to generate checkpoints.</p>
      </div>
    );
  }

  return (
    <div className="checkpoint-selector">
      <div className="checkpoint-header">
        <h4 className="checkpoint-title">
          <Layers size={16} /> Select Candidate Model Checkpoint
        </h4>
        <span className="checkpoint-count">{checkpoints.length} Checkpoints Available</span>
      </div>

      <div className="checkpoints-list">
        {checkpoints.map((cp) => {
          const isSelected = cp.name === selectedCheckpoint;
          return (
            <div
              key={cp.name}
              className={`checkpoint-item ${isSelected ? 'selected' : ''}`}
              onClick={() => onSelectCheckpoint(cp.name)}
            >
              <div className="checkpoint-info">
                <div className="checkpoint-name-row">
                  <span className="name">{cp.name}</span>
                  {cp.is_best && (
                    <span className="best-pill">
                      <Award size={12} /> checkpoint_best
                    </span>
                  )}
                </div>
                <span className="step-label">
                  {cp.step > 0 ? `Step ${cp.step.toLocaleString()}` : 'Best Evaluated'}
                </span>
              </div>

              <div className="checkpoint-select-radio">
                {isSelected ? <CheckCircle size={18} className="text-primary" /> : <div className="radio-circle" />}
              </div>
            </div>
          );
        })}
      </div>

      {selectedCheckpoint === 'checkpoint_best' && (
        <div className="warning-note">
          <AlertTriangle size={14} />
          <span>
            <strong>Caution:</strong> checkpoint_best can sometimes over-index on noisy early value-sensitivity. Consider picking the highest step or verifying with a manual sweep.
          </span>
        </div>
      )}
    </div>
  );
};
