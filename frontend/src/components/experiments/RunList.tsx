import React, { useState } from 'react';
import { Search, Filter, Database } from 'lucide-react';
import { Run } from '../../types';
import { RunCard } from './RunCard';

interface RunListProps {
  runs: Run[];
  selectedRunId: string | null;
  onSelectRun: (runId: string) => void;
  onInspectRepair: (runId: string) => void;
  onToggleArchive: (runId: string) => void;
  onDeleteRun: (runId: string) => void;
  showArchived?: boolean;
}

export const RunList: React.FC<RunListProps> = ({
  runs,
  selectedRunId,
  onSelectRun,
  onInspectRepair,
  onToggleArchive,
  onDeleteRun,
  showArchived = false,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [agentFilter, setAgentFilter] = useState<'all' | '2' | '3'>('all');

  const filteredRuns = runs.filter((run) => {
    if (showArchived ? !run.archived : run.archived) return false;
    
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      const matchName = run.run_id.toLowerCase().includes(term);
      const matchExp = run.experiment_name.toLowerCase().includes(term);
      if (!matchName && !matchExp) return false;
    }

    if (agentFilter !== 'all') {
      const count = (run.config?.num_agents || 2).toString();
      if (count !== agentFilter) return false;
    }

    return true;
  });

  return (
    <div className="run-list-container">
      {/* Filter and Search Bar */}
      <div className="list-controls">
        <div className="search-bar">
          <Search size={16} />
          <input
            type="text"
            placeholder="Search experiments or run IDs..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        <div className="filter-group">
          <Filter size={15} />
          <span>Agents:</span>
          <select
            value={agentFilter}
            onChange={(e) => setAgentFilter(e.target.value as any)}
            className="filter-select"
          >
            <option value="all">All Counts</option>
            <option value="2">2 Agents</option>
            <option value="3">3 Agents</option>
          </select>
        </div>
      </div>

      {/* Grid of Runs */}
      {filteredRuns.length === 0 ? (
        <div className="empty-state">
          <Database size={36} className="text-gray" />
          <h4>No {showArchived ? 'archived' : 'active'} experiments found</h4>
          <p>Launch a new training experiment from the Simulate tab.</p>
        </div>
      ) : (
        <div className="runs-grid">
          {filteredRuns.map((run) => (
            <RunCard
              key={run.run_id}
              run={run}
              isSelected={run.run_id === selectedRunId}
              onSelect={onSelectRun}
              onInspectRepair={onInspectRepair}
              onToggleArchive={onToggleArchive}
              onDelete={onDeleteRun}
            />
          ))}
        </div>
      )}
    </div>
  );
};
