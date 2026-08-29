import React from 'react';
import { PlaySquare, ListFilter } from 'lucide-react';
import { Run, RunConfig } from '../types';
import { ExperimentLauncher } from '../components/experiments/ExperimentLauncher';
import { RunList } from '../components/experiments/RunList';

interface ExperimentsPageProps {
  runs: Run[];
  selectedRunId: string | null;
  onLaunch: (config: RunConfig) => Promise<void>;
  isLaunching: boolean;
  isBusy: boolean;
  onSelectRun: (runId: string) => void;
  onInspectRepair: (runId: string) => void;
  onToggleArchive: (runId: string) => void;
  onDeleteRun: (runId: string) => void;
}

export const ExperimentsPage: React.FC<ExperimentsPageProps> = ({
  runs,
  selectedRunId,
  onLaunch,
  isLaunching,
  isBusy,
  onSelectRun,
  onInspectRepair,
  onToggleArchive,
  onDeleteRun,
}) => {
  return (
    <div className="page-container">
      <div className="page-header">
        <div>
          <h2>
            <PlaySquare className="inline mr-2 text-primary" size={22} /> Simulate & Train Experiments
          </h2>
          <p>
            Configure environment parameters, set agent and landmark counts, and train emergent communication policies.
          </p>
        </div>
      </div>

      {/* Launcher Form */}
      <div className="launcher-card-wrapper">
        <ExperimentLauncher
          onLaunch={onLaunch}
          isLaunching={isLaunching}
          isBusy={isBusy}
        />
      </div>

      {/* Experiments History */}
      <div className="section-container mt-6">
        <div className="section-header">
          <h3>
            <ListFilter size={18} className="inline mr-2 text-gray" /> All Experiments
          </h3>
        </div>
        <RunList
          runs={runs}
          selectedRunId={selectedRunId}
          onSelectRun={onSelectRun}
          onInspectRepair={onInspectRepair}
          onToggleArchive={onToggleArchive}
          onDeleteRun={onDeleteRun}
          showArchived={false}
        />
      </div>
    </div>
  );
};
